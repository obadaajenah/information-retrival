from save_read import load_dataset ,load_dataset2
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np

def sanitize_data(data):
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None  # يمكنك استبدالها بقيمة افتراضية إذا لزم الأمر
    return data



def extract_doc_ids(query_id, df):
    query_id = int(query_id)
    # قم بتصفية DataFrame باستخدام رقم الاستعلام وشرط relevance
    filtered_df = df[(df['query_id'] == query_id) & ((df['relevance'] == 1) | (df['relevance'] == 2))]
    
    # استخراج قيم doc_id وتحويلها إلى قائمة
    doc_ids = filtered_df['doc_id'].tolist()
    relevent =filtered_df['relevance'].tolist()
    return doc_ids , relevent


    
# Get documents for query function
def get_documents_for_query(query, tfidf_matrix, processor, vectorizer, data):
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
    n = 10
    top_documents_indices = cosine_similarities.argsort()[-n:][::-1]
    top_documents = data.iloc[top_documents_indices]
    return top_documents, cosine_similarities[top_documents_indices]






def get_documents_for_query2(query: str, data_vector_file: str, vectorizer_file: str, data_file: str):
        try:
            print("dddddddddddddddd")
            with open(data_vector_file, "rb") as file:
                data_vector = pickle.load(file)
            print("ffffffffffffffffffffff")
            with open(vectorizer_file, "rb") as file:
                vectorizer = pickle.load(file)
            print("pppppppppppppppp")
            data = load_dataset2(data_file)
            print("eeeeeeeeeeeeeeeeeeeeeeeeee")
            query_vector = vectorizer.transform([query])
            print("zzzzzzzzzzzzzzzzzzzzzzz")
            cosine_similarities = cosine_similarity(data_vector, query_vector).flatten().tolist()
            print("hhhhhhhhhhhhhhhhhhhhhhh")
            cosine_similarities = [sanitize_data(x) for x in cosine_similarities]
            print("llllllllllllllllllllllllllllllll")
            cosine_similarities = [int(x) for x in cosine_similarities]
            n = 10
            top_documents_indices = np.argsort(cosine_similarities)[-n:][::-1]
            print("ttttttttttttttttttttttttttttttttttttt")
            top_documents = data.iloc[top_documents_indices]
            print("sssssssssssssssssssssssssssssssssssssssssssssssss")

            return top_documents , cosine_similarities[top_documents_indices]
         
        except Exception as e:
            print(f"Error occurred: {e}")
            return {"error": str(e)}