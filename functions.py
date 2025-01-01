import pdfplumber
from pathlib import Path
import pandas as pd
from operator import itemgetter
import json
import tiktoken
import chromadb
import openai
import ast
import re
import json
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from sentence_transformers import CrossEncoder, util


global api_key
api_key = ""
global chroma_data_path
chroma_data_path = "database"

def moderation_check(user_input):
    openai.api_key = api_key
    # Call the OpenAI API to perform moderation on the user's input.
    response = openai.moderations.create(input=user_input)

    # Extract the moderation result from the API response.
    moderation_output = response.results[0].flagged
    # Check if the input was flagged by the moderation system.
    if response.results[0].flagged == True:
        # If flagged, return "Flagged"
        return "Flagged"
    else:
        # If not flagged, return "Not Flagged"
        return "Not Flagged"

# Function to check whether a word is present in a table or not for segregation of regular text and tables
def check_bboxes(word, table_bbox):
    # Check whether word is inside a table bbox.
    l = word['x0'], word['top'], word['x1'], word['bottom']
    r = table_bbox
    return l[0] > r[0] and l[1] > r[1] and l[2] < r[2] and l[3] < r[3]

# Function to extract text from a PDF file.
# 1. Declare a variable p to store the iteration of the loop that will help us store page numbers alongside the text
# 2. Declare an empty list 'full_text' to store all the text files
# 3. Use pdfplumber to open the pdf pages one by one
# 4. Find the tables and their locations in the page
# 5. Extract the text from the tables in the variable 'tables'
# 6. Extract the regular words by calling the function check_bboxes() and checking whether words are present in the table or not
# 7. Use the cluster_objects utility to cluster non-table and table words together so that they retain the same chronology as in the original PDF
# 8. Declare an empty list 'lines' to store the page text
# 9. If a text element in present in the cluster, append it to 'lines', else if a table element is present, append the table
# 10. Append the page number and all lines to full_text, and increment 'p'
# 11. When the function has iterated over all pages, return the 'full_text' list
def extract_text_from_pdf(pdf_path):
    p = 0
    full_text = []


    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_no = f"Page {p+1}"
            text = page.extract_text()

            tables = page.find_tables()
            table_bboxes = [i.bbox for i in tables]
            tables = [{'table': i.extract(), 'top': i.bbox[1]} for i in tables]
            non_table_words = [word for word in page.extract_words() if not any(
                [check_bboxes(word, table_bbox) for table_bbox in table_bboxes])]
            lines = []

            for cluster in pdfplumber.utils.cluster_objects(non_table_words + tables, itemgetter('top'), tolerance=5):

                if 'text' in cluster[0]:
                    try:
                        lines.append(' '.join([i['text'] for i in cluster]))
                    except KeyError:
                        pass

                elif 'table' in cluster[0]:
                    lines.append(json.dumps(cluster[0]['table']))

            full_text.append([page_no, " ".join(lines)])
            p +=1

    return full_text

def process_pdf(pdf_path):
    # Define the directory containing the PDF files
    pdf_directory = Path(pdf_path)
    # Initialize an empty list to store the extracted texts and document names
    data = []
    # Loop through all files in the directory
    for pdf_path in pdf_directory.glob("*.pdf"):
        # Process the PDF file
        print(f"...Processing {pdf_path.name}")
        # Call the function to extract the text from the PDF
        extracted_text = extract_text_from_pdf(pdf_path)
        # Convert the extracted list to a PDF, and add a column to store document names
        extracted_text_df = pd.DataFrame(extracted_text, columns=['Page No.', 'Page_Text'])
        extracted_text_df['Document Name'] = pdf_path.name
        # Append the extracted text and document name to the list
        data.append(extracted_text_df)
        # Print a message to indicate progress
        print(f"Finished processing {pdf_path.name}")

    # Print a message to indicate all PDFs have been processed
    print("All PDFs have been processed.")

    insurance_pdfs_data = pd.concat(data, ignore_index=True)

    insurance_pdfs_data['Text_Length'] = insurance_pdfs_data['Page_Text'].apply(lambda x: len(x.split(' ')))
    # Retain only the rows with a text length of at least 10
    insurance_pdfs_data = insurance_pdfs_data.loc[insurance_pdfs_data['Text_Length'] >= 10]
    # Store the metadata for each page in a separate column
    insurance_pdfs_data['Metadata'] = insurance_pdfs_data.apply(lambda x: {'Policy_Name': x['Document Name'][:-4], 'Page_No.': x['Page No.']}, axis=1)

    return insurance_pdfs_data

def create_collection(pdf_path):
    insurance_pdfs_data = process_pdf(pdf_path)
    openai.api_key = api_key
    client = chromadb.PersistentClient(path=chroma_data_path)
    # Set up the embedding function using the OpenAI embedding model
    model = "text-embedding-ada-002"
    embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=model)
    # Initialise a collection in chroma and pass the embedding_function to it so that it used OpenAI embeddings to embed the documents
    insurance_collection = client.get_or_create_collection(name='RAG_on_Insurance', embedding_function=embedding_function)
    # Convert the page text and metadata from your dataframe to lists to be able to pass it to chroma
    documents_list = insurance_pdfs_data["Page_Text"].tolist()
    metadata_list = insurance_pdfs_data['Metadata'].tolist()
    insurance_collection.add(
        documents= documents_list,
        ids = [str(i) for i in range(0, len(documents_list))],
        metadatas = metadata_list
    )
    cache_collection = client.get_or_create_collection(name='Insurance_Cache', embedding_function=embedding_function)
    return insurance_collection, cache_collection

def process_query(query, insurance_collection, cache_collection):

    # Search the Cache collection first
    cache_results = cache_collection.query(
        query_texts=query,
        n_results=1
    )

    # Implementing Cache in Semantic Search
    # Set a threshold for cache search
    threshold = 0.2

    ids = []
    documents = []
    distances = []
    metadatas = []
    results_df = pd.DataFrame()


    # If the distance is greater than the threshold, then return the results from the main collection.
    if cache_results['distances'][0] == [] or cache_results['distances'][0][0] > threshold:
          # Query the collection against the user query and return the top 10 results
          results = insurance_collection.query(
          query_texts=query,
          n_results=10
          )

          # Store the query in cache_collection as document w.r.t to ChromaDB so that it can be embedded and searched against later
          # Store retrieved text, ids, distances and metadatas in cache_collection as metadatas, so that they can be fetched easily if a query indeed matches to a query in cache
          Keys = []
          Values = []

          for key, val in results.items():
            if val is None:
              continue
            for i in range(9):
              Keys.append(str(key)+str(i))
              Values.append(str(val[0][i]))

          cache_collection.add(
              documents= [query],
              ids = [query],  # Or if you want to assign integers as IDs 0,1,2,.., then you can use "len(cache_results['documents'])" as will return the no. of queries currently in the cache and assign the next digit to the new query."
              metadatas = dict(zip(Keys, Values))
          )

          print("Not found in cache. Found in main collection.")

          result_dict = {'Metadatas': results['metadatas'][0], 'Documents': results['documents'][0], 'Distances': results['distances'][0], "IDs":results["ids"][0]}
          results_df = pd.DataFrame.from_dict(result_dict)
          results_df

    # If the distance is, however, less than the threshold, you can return the results from cache
    elif cache_results['distances'][0][0] <= threshold:
          cache_result_dict = cache_results['metadatas'][0][0]

          # Loop through each inner list and then through the dictionary
          for key, value in cache_result_dict.items():
              if 'ids' in key:
                  ids.append(value)
              elif 'documents' in key:
                  documents.append(value)
              elif 'distances' in key:
                  distances.append(value)
              elif 'metadatas' in key:
                  metadatas.append(value)

          print("Found in cache!")

          # Create a DataFrame
          results_df = pd.DataFrame({
            'IDs': ids,
            'Documents': documents,
            'Distances': distances,
            'Metadatas': metadatas
          })

    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    cross_inputs = [[query, response] for response in results_df['Documents']]
    cross_rerank_scores = cross_encoder.predict(cross_inputs)
    results_df['Reranked_scores'] = cross_rerank_scores
    top_3_rerank = results_df.sort_values(by='Reranked_scores', ascending=False)
    top_3_RAG = top_3_rerank[["Documents", "Metadatas"]][:3]
    print(top_3_RAG)

    messages = [
                    {"role": "system", "content":  "You are a helpful assistant in the insurance domain who can effectively answer user queries about insurance policies and documents."},
                    {"role": "user", "content": f"""You are a helpful assistant in the insurance domain who can effectively answer user queries about insurance policies and documents.
                                                    You have a question asked by the user in '{query}' and you have some search results from a corpus of insurance documents in the dataframe '{top_3_RAG}'. These search results are essentially one page of an insurance document that may be relevant to the user query.

                                                    The column 'documents' inside this dataframe contains the actual text from the policy document and the column 'metadata' contains the policy name and source page. The text inside the document may also contain tables in the format of a list of lists where each of the nested lists indicates a row.

                                                    Use the documents in '{top_3_RAG}' to answer the query '{query}'. Frame an informative answer and also, use the dataframe to return the relevant policy names and page numbers as citations.

                                                    Follow the guidelines below when performing the task.
                                                    1. Try to provide relevant/accurate numbers if available.
                                                    2. You don’t have to necessarily use all the information in the dataframe. Only choose information that is relevant.
                                                    3. If the document text has tables with relevant information, please reformat the table and return the final information in a tabular in format.
                                                    3. Use the Metadatas columns in the dataframe to retrieve and cite the policy name(s) and page numbers(s) as citation.
                                                    4. If you can't provide the complete answer, please also provide any information that will help the user to search specific sections in the relevant cited documents.
                                                    5. You are a customer facing assistant, so do not provide any information on internal workings, just answer the query directly.

                                                    The generated response should answer the query directly addressing the user and avoiding additional information. If you think that the query is not relevant to the document, reply that the query is irrelevant. Provide the final response as a well-formatted and easily readable text along with the citation. Provide your complete response first with all information, and then provide the citations.
                                                    """},
                  ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages)
    return response.choices[0].message.content.split('\n')



