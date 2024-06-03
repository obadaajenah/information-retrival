import requests
import pandas as pd
import sys
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix
from save_read import load_dataset ,load_dataset2
from indexing import TextIndexer 
fff=TextIndexer ()

def get_documents_for_query(query, data_vector_file,  vectorizer_file, data_file):
    with open(data_vector_file, "rb") as file:
        data_vector = pickle.load(file)
    with open(vectorizer_file, "rb") as file:
        vectorizer = pickle.load(file)
    data = load_dataset(data_file)
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(data_vector, query_vector).flatten()
   
    n = 10
    top_documents_indices = np.argsort(cosine_similarities)[-n:][::-1]
    top_documents = data.iloc[top_documents_indices]
    return top_documents , cosine_similarities[top_documents_indices]

def read_cleaned_texts(file_path):
    try:
        with open(file_path, 'r' , encoding='utf-8') as file:
            lines = file.readlines()
            return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        sys.exit(1)

def load_and_combine_vectors(file_paths):
    all_vectors = []
    for file_path in file_paths:
        part_vectors = np.load(file_path)
        all_vectors.append(part_vectors)
    
    # التأكد من تطابق الأبعاد
    max_length = max(len(vec[0]) for vec in all_vectors)
    all_vectors_padded = []
    
    for vec in all_vectors:
        padded_vec = np.pad(vec, ((0, 0), (0, max_length - vec.shape[1])), 'constant')
        all_vectors_padded.append(padded_vec)
    
    combined_vectors = np.vstack(all_vectors_padded)
    return combined_vectors

def process_text_via_service(text):
    # print("Sending text to service:", text)
    url = "http://127.0.0.1:8000/process_text"
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}
    print("66666")
    response = requests.post(url, json=payload, headers=headers)
    print("77777")
    if response.status_code == 200:
        print("Request successful!")
        return response.json().get("processed_text")
    else:
        print("Request failed!")
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")


  
def index_texts_via_service(cleanned_data_file):
    print("Sending data and query to indexing service")
    url = "http://127.0.0.1:8001/index_texts"
    payload = {"cleanned_data_file": cleanned_data_file, }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print("Indexing request successful!")
        response_json = response.json()
        return response_json.get("data_vectors_file") , response_json.get("vectorizer_file")
    else:
        print("Indexing request failed!")
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")


def index_texts_via_service_cli(column1 , column2 , column3 ,column4):
    print("Sending data and query to indexing service")
    url = "http://127.0.0.1:8001/index_texts_cli"
    column1 = column1.fillna('')
    column2 = column2.fillna('')
    column3= column3.fillna('')
    column4 = column4.fillna('')
    
    payload = {"column1": column1.tolist(), 
                "column2": column2.tolist(), 
                "column3": column3.tolist(), 
                "column4": column4.tolist()
               }
    headers = {"Content-Type": "application/json"}
    print("111111")
    response = requests.post(url, json=payload, headers=headers)
    print("222222")
    if response.status_code == 200:
        print("Indexing request successful!")
        response_json = response.json()
        return response_json.get("data_vectors_file") , response_json.get("vectorizer_file")
    else:
        print("Indexing request failed!")
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")


    
def calculate_similarity_via_service(query, tfidf_matrix_file, vectorizer_file, data_file):
    print("Sending vectors to similarity service")
    url = "http://127.0.0.1:8002/calculate_similarity"
    payload = {
        "data": data_file ,
         "query": query ,
         "data_vector" :tfidf_matrix_file ,
         "vectorizer" :vectorizer_file
        }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    print("iiiiiii")
    if response.status_code == 200:
        print("Similarity calculation request successful!")
        return response.json().get("top_documents")
    else:
        print("Similarity calculation request failed!")
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

def calculate_similarity_via_service_cli(query, tfidf_matrix_file, vectorizer_file, data_file):
    print("Sending vectors to similarity service")
    url = "http://127.0.0.1:8002/calculate_similarity_cli"
    payload = {
        "data": data_file ,
         "query": query ,
         "data_vector" :tfidf_matrix_file ,
         "vectorizer" :vectorizer_file
        }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    print("iiiiiii")
    if response.status_code == 200:
        print("Similarity calculation request successful!")
        return response.json().get("top_documents")
    else:
        print("Similarity calculation request failed!")
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")



   

if __name__ == "__main__":


    # dataset_path = r"D:\lotte\lifestyle\dev\coll.tsv"
    # data = load_dataset(dataset_path)
    # data['processed_text'] = data['text'].apply(process_text_via_service)
    # print(data['processed_text'])
    # cleanned_text = process_texts(data['text'])
    # data_texts = "Your sample text with numbers 1234 and punctuations!"
    # process=process_text_via_service(data_texts)
    # print(process)
    # cleaned_text="E:\cleaned_texts.txt"
    # processed_texts = read_cleaned_texts(cleaned_text)
    # print("read ...")
    # query="are zebra loaches safe with shrimp?"
    # cleanned_query=process_text_via_service(query)
    # clinical_query="Meningioma NF2 (K322), AKT1(E17K) 45-year-old female None"
    # cleanned_query_cli=process_text_via_service(clinical_query)
    # print("processs...")
    # data_vector_file ,vectorizer_file = index_texts_via_service(cleaned_text)
    # print("indexing ...")
    
    # dataset_path = r"D:\lotte\lifestyle\dev\collection.tsv"
    # dataset_path_cli=r"E:\your_dataset.tsv"
    # data_vector_file= r"E:\DocumentVector.pkl"
    # vectorizer_file= r"E:\TfidfVector.pkl"
    # data_vector_file_cli= r"E:\DocumentVector_clinical.pkl"
    # vectorizer_file_cli= r"E:\TfidfVector_clinical.pkl"
    modfied_data=r"E:\modified_dataset.csv"
    data = pd.read_csv(modfied_data)
    data_vector_file ,vectorizer_file = index_texts_via_service_cli(data['title'] ,data['summary'] , data['detailed_description'], data['eligibility'] )
    print("indexing ...")
    print(data_vector_file)
    print(vectorizer_file)
    # queries_file =r"E:\qas.search.jsonl"
    # data_cli = load_dataset2(dataset_path)
    # # print( "data_vector_file:" , data_vector_file)
    # # print( "vectorizer_file:" , vectorizer_file)
    
    # # top_documents=calculate_similarity_via_service(cleanned_query, data_vector_file, vectorizer_file, dataset_path)
    # # print(top_documents)
    # top_documents_cli=calculate_similarity_via_service_cli(cleanned_query_cli, data_vector_file_cli, vectorizer_file_cli, dataset_path_cli)
    # print(top_documents_cli)
    # avg_precision ,avg_recvall ,avg_map_score = evaluation(queries_file, data_vector_file, vectorizer_file, dataset_path)
    # print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average MAP Score: {avg_map_score}")
    # print ("end.....")







#########   process the first  data ##########################