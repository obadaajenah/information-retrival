
from fastapi import FastAPI, Request
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from save_read import load_dataset ,read_cleaned_texts ,save_vectors_to_binary_file
from fastapi import FastAPI, Request
import uvicorn

# كلاس لتحويل النصوص إلى متجهات
class TextIndexer:
 

 def clean_space(column1 , column2,column3,column4):
        column1 = column1.fillna('')
        column2 = column2.fillna('')
        column3= column3.fillna('')
        column4 = column4.fillna('')
        return column1,column2 ,column3 ,column4


# إعداد FastAPI
app = FastAPI()

# إنشاء كائن من TextIndexer
indexer = TextIndexer()
###################    The first dataset ############################33333333

@app.post("/index_texts")
async def index_texts(request: Request):
    try:
        print("start")
        # استخراج البيانات من الطلب
        data = await request.json()
        
        data_file = data.get("cleanned_data_file")

        data_texts = read_cleaned_texts(data_file)
         
        if  data_texts is None :
            return {"error": " 'data' is fields are required."}
       
        vectorizer = TfidfVectorizer()
        # تحويل النصوص إلى متجهات
        data_vectors = vectorizer.fit_transform( data_texts)
       
        data_vector_file=r"D:\DocumentVector.pkl"
        vectorizer_file= r"D:\TfidfVector.pkl"
        save_vectors_to_binary_file(data_vectors ,data_vector_file)
        
        save_vectors_to_binary_file(vectorizer ,vectorizer_file)
        
        return {
            "data_vectors_file": data_vector_file,
            "vectorizer_file": vectorizer_file,
            
        }
    except Exception as e:
        return {"error": str(e)}


################# The secound dataset ########################

@app.post("/index_texts_cli")
async def index_texts(request: Request):
    try:
       
        print("start")
        # استخراج البيانات من الطلب
        data = await request.json()
    
        # print(data)
        column1 = data.get("column1")
        column2 = data.get("column2")
        column3 = data.get("column3")
        column4 = data.get("column4")
        
        vectorizer = TfidfVectorizer()
       
        column1_vec = vectorizer.fit_transform(column1)
        column2_vec = vectorizer.fit_transform(column2)
        column3_vec = vectorizer.fit_transform(column3)
        column4_vec = vectorizer.fit_transform(column4)

        combined_vocabulary = set(vectorizer.vocabulary_.keys())
      
        # إعادة بناء vectorizer بناءً على المفردات المدمجة
        combined_vocabulary = list(combined_vocabulary)
        combined_vectorizer = TfidfVectorizer(vocabulary=combined_vocabulary)
    
        # إعادة تحويل النصوص باستخدام الـ vectorizer المدمج
        column1_vec = combined_vectorizer.fit_transform(column1)
        column2_vec = combined_vectorizer.fit_transform(column2)
        column3_vec = combined_vectorizer.fit_transform(column3)
        column4_vec = combined_vectorizer.fit_transform(column4)
        
        col1_weight = 4.0  
        col4_weight = 1.5
        # حساب المتوسط المرجح
        weighted_col1_tfidf = col1_weight * column1_vec
       
        weighted_col4_tfidf = col4_weight * column4_vec
        # جمع المصفوفات
        sum_tfidf = weighted_col1_tfidf + column2_vec + weighted_col4_tfidf + column4_vec

        # حساب المتوسط المرجح
        average_tfidf = sum_tfidf / (col1_weight + 1 + col4_weight + 1)
       
              
        data_vector_file=r"D:\DocumentVector_clinical.pkl"
        vectorizer_file= r"D:\TfidfVector_clinical.pkl"

        save_vectors_to_binary_file(average_tfidf ,data_vector_file)
       
        save_vectors_to_binary_file(combined_vectorizer ,vectorizer_file)
        
        return {
            "data_vectors_file": data_vector_file,
            "vectorizer_file": vectorizer_file,
            
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
