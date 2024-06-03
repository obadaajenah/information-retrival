from fastapi import FastAPI, Request
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from save_read import load_dataset , load_dataset2
import uvicorn
import math

class SimilarityCalculator:
    
    def sanitize_data(data):
        if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
            return None  # يمكنك استبدالها بقيمة افتراضية إذا لزم الأمر
        return data
    

    @staticmethod
    def get_documents_for_query(query, data_vector_file,  vectorizer_file, data_file):
        try:
           
            with open(data_vector_file, "rb") as file:
                data_vector = pickle.load(file)
          
            with open(vectorizer_file, "rb") as file:
                vectorizer = pickle.load(file)

            data = load_dataset(data_file)

            query_vector = vectorizer.transform([query])

            cosine_similarities = cosine_similarity(data_vector, query_vector).flatten().tolist()

            cosine_similarities = [int(x) for x in cosine_similarities]
            
            n = 10

            top_documents_indices = np.argsort(cosine_similarities)[-n:][::-1]

            top_documents = data.iloc[top_documents_indices]

            return top_documents , cosine_similarities
        
        except Exception as e:
            print(f"Error occurred: {e}")
            return {"error": str(e)}
        


    @staticmethod
    def get_documents_for_query2(query: str, data_vector_file: str, vectorizer_file: str, data_file: str):
        try:
            
            with open(data_vector_file, "rb") as file:
                data_vector = pickle.load(file)
           
            with open(vectorizer_file, "rb") as file:
                vectorizer = pickle.load(file)
            
            data = load_dataset2(data_file)
            
            query_vector = vectorizer.transform([query])
          
            cosine_similarities = cosine_similarity(data_vector, query_vector).flatten().tolist()
            
            cosine_similarities = [SimilarityCalculator.sanitize_data(x) for x in cosine_similarities]
            
            n = 10
            top_documents_indices = np.argsort(cosine_similarities)[-n:][::-1]
            
            top_documents = data.iloc[top_documents_indices]
            
            return top_documents , cosine_similarities
         
        except Exception as e:
            print(f"Error occurred: {e}")
            return {"error": str(e)}

   


app = FastAPI()

@app.post("/calculate_similarity")
async def calculate_similarity(request: Request):
    res = await request.json()
    data_file = res.get("data")
    query = res.get("query",)
    data_vector_file= res.get("data_vector")
    vectorizer_file= res.get("vectorizer")
    top_documents ,cos_similarities = SimilarityCalculator.get_documents_for_query(query, data_vector_file,  vectorizer_file, data_file)
    top_documents_json = top_documents.to_dict(orient='records')
    return {  "top_documents":top_documents_json,}


@app.post("/calculate_similarity_cli")
async def calculate_similarity_cli(request: Request):
    res = await request.json()
    data_file = res.get("data")
    query = res.get("query",)
    data_vector_file= res.get("data_vector")
    vectorizer_file= res.get("vectorizer")
    top_documents = SimilarityCalculator.get_documents_for_query2(query, data_vector_file,  vectorizer_file, data_file)
 
    # if isinstance(top_documents, dict) and "error" in top_documents:
    #     print("isinstance....")
    #     return top_documents
  
    top_documents = top_documents.fillna('')
    top_documents_json = top_documents.to_dict(orient='records')
    

    return {"top_documents": top_documents_json}




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
    