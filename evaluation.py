from fastapi import FastAPI, Request
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import pandas as pd
import pickle
import json
from save_read import load_dataset
from sklearn.metrics import precision_score, recall_score, average_precision_score
from TextProcessing import TextProcessor, processtext
import uvicorn
from helper import extract_doc_ids
from main import get_documents_for_query 
from helper import get_documents_for_query2
from save_read import load_dataset2 , read_csv_to_array

class Evaluation:
 
    @staticmethod
    def calculate_precision_recall(y_true, y_pred, threshold=0.5):
        y_pred_binary = (y_pred >= threshold).astype(int)
        precision = precision_score(y_true, y_pred_binary, average='micro')
        recall = recall_score(y_true, y_pred_binary, average='micro')
        return precision, recall

    @staticmethod
    def calculate_map_score(y_true, y_pred):
        return average_precision_score(y_true, y_pred, average='micro')
    
    # def calculate_mrr(y_true):
    #      # Calculate reciprocal rank
           
    @staticmethod
    def load_queries(queries_paths):
        queries = []
        for file_path in queries_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        query = json.loads(line.strip())
                        if 'query' in query:
                            queries.append(query)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid line in {file_path}: {line}")
        return queries
    
    @staticmethod
    def process_texts(texts):
        processed_texts = []
        for text in texts:
            #print("text: " + text)
            processed_text = processtext(text)
            processed_texts.append(processed_text)
        return processed_texts
    
    def read_csv_to_array(file_path):
        # قراءة ملف CSV باستخدام pandas
        df = pd.read_csv(file_path)
        
        # تحويل إطار البيانات إلى مصفوفة من القوائم
        data_array = df.values.tolist()
        
        return data_array

    @staticmethod
    def get_documents_for_query_cli(query, tfidf_matrix, vectorizer, data):
        with open(tfidf_matrix, "rb") as file:
            tfidf_matrix = pickle.load(file)
            
        with open(vectorizer, "rb") as file:
            vectorizer = pickle.load(file)
            
        data = load_dataset2(data)
            
        query_vector = vectorizer.transform([query])
        cosine_similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
        n = 10
        top_documents_indices = cosine_similarities.argsort()[-n:][::-1]
        top_documents = data.iloc[top_documents_indices]
    
        return top_documents, cosine_similarities[top_documents_indices]

    @staticmethod
    def evaluation(queries_file, tfidf_matrix_file, vectorizer_file, data_file):
            queries = Evaluation.load_queries([queries_file])
            all_precisions = []
            all_recalls = []
            all_map_scores = []
            all_rr = []
            
            for query in queries:
                if 'query' in query:
                    processed_query = Evaluation.process_texts([query['query']])[0]
                    print(processed_query)
                    processed_query = processed_query['processed_text']
                    top_documents, cosine_similarities = get_documents_for_query(processed_query, tfidf_matrix_file, vectorizer_file, data_file)
                    
                    data = load_dataset(data_file)
                    relevance = np.zeros(len(data))
                    for pid in query.get('answer_pids', []):
                        relevance[np.where(data['pid'] == pid)[0]] = 1

                    y_true = relevance[top_documents.index]
                    y_pred = cosine_similarities

                    if y_true.sum() == 0:
                        print(f"No relevant documents for query ID: {query.get('qid', 'N/A')}")
                        # print("Documents found:", top_documents)
                        continue

                    precision, recall = Evaluation.calculate_precision_recall(y_true, y_pred)
                    all_precisions.append(precision)
                    all_recalls.append(recall)

                    map_score = Evaluation.calculate_map_score(y_true, y_pred)
                    all_map_scores.append(map_score)
                    # Calculate reciprocal rank
                    rr = 0
                    for i, rel in enumerate(y_true):
                        if rel == 1:
                            rr = 1 / (i + 1)  # Reciprocal rank
                            break
                    all_rr.append(rr)
                    print(f"Query ID: {query.get('qid', 'N/A')}, Precision: {precision}, Recall: {recall}, MAP Score: {map_score}, RR: {rr}")
                    # print(
                    #     f"Query ID: {query.get('qid', 'N/A')}, Precision: {precision}, Recall: {recall}, MAP Score: {map_score}, RR: {rr}")

        
            avg_precision = np.mean(all_precisions)
            avg_recall = np.mean(all_recalls)
            avg_map_score = np.mean(all_map_scores)
            mrr = np.mean(all_rr)
            # print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average MAP Score: {avg_map_score}, MRR: {mrr}")
            # return avg_precision ,avg_recall ,avg_map_score
            print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average MAP Score: {avg_map_score}, MRR: {mrr}")       

    @staticmethod
    def evaluation_clic(queries_file, tfidf_matrix_file, vectorizer_file, data_file):
            print("111111111111111")
            queries = read_csv_to_array(queries_file)
            all_precisions = []
            all_recalls = []
            all_map_scores = []
            all_rr = []
            print("22222222222")
            for query in queries:
                query = [str(item) for item in query]
                combined_query = ' '.join(query)
                id=query[0]  
                print("333333")
                top_documents, cosine_similarities = Evaluation.get_documents_for_query_cli(combined_query, tfidf_matrix_file,vectorizer_file, data_file)
                qrles=r"D:\qrels.csv"
                df = pd.read_csv(qrles)
                doc_ids , relevence = extract_doc_ids(id ,df)
                modfied_data=r"D:\modified_dataset.csv"
                data = pd.read_csv(modfied_data)
                relevance = np.zeros(len(data))
                for doc_id in doc_ids:
                        relevance[np.where(data['doc_id'] == doc_id)[0]] = 1

                y_true = relevance[top_documents.index]
                y_pred = cosine_similarities

                if y_true.sum() == 0:
                        print(f"No relevant documents for query ID: {query[0] }")
                        continue

                precision, recall =Evaluation.calculate_precision_recall(y_true, y_pred)
                all_precisions.append(precision)
                all_recalls.append(recall)

                map_score = Evaluation.calculate_map_score(y_true, y_pred)
                all_map_scores.append(map_score)
                rr = 0
                for i, rel in enumerate(y_true):
                    if rel == 1:
                        rr = 1 / (i + 1)  # Reciprocal rank
                        break
                all_rr.append(rr)

                print(f"Query ID: {query[0] }, Precision: {precision}, Recall: {recall}, MAP Score: {map_score}, RR: {rr}")

            avg_precision = np.mean(all_precisions)
            avg_recall = np.mean(all_recalls)
            avg_map_score = np.mean(all_map_scores)
            mrr = np.mean(all_rr)

            print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average MAP Score: {avg_map_score}, MRR: {mrr}")

        


evaluation =Evaluation()
###############  Evaluation the first dataset ###########################

# dataset_path = r"D:\collection.tsv"

# data_vector_file= r"D:\DocumentVector.pkl"
# vectorizer_file= r"D:\TfidfVector.pkl"
# queries_file =r"C:\Users\USER\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.forum.jsonl"

# evaluation.evaluation(queries_file ,data_vector_file,vectorizer_file,dataset_path )
    
####################### Evaluation the secound  dataset   #########################


yourdataset=r"D:\your_dataset.csv"


process_queries=r"D:\processed_queries.csv"


data_vector_file_cli= r"D:\DocumentVector_clinical.pkl"

vectorizer_file_cli= r"D:\TfidfVector_clinical.pkl"  

Evaluation.evaluation_clic(process_queries,data_vector_file_cli,vectorizer_file_cli,yourdataset)