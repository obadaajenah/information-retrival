from flask import Flask, render_template, request, jsonify
from helper import get_documents_for_query
from TextProcessing import TextProcessor, processtext
from Query_Suggestion import Query_Suggestion 
from Query_Refinement import Query_Refinement
import nltk
import json
import pandas as pd
import pickle
from  save_read import load_dataset2, load_dataset , read_cleaned_texts
processor = TextProcessor()
query_suggester = None
query_refirment = None
dataset_path = ''
dataset=None
data = None
cleaned_texts = None
cleaned_texts2 = None
tfidf_matrix = None
vectorizer = None

# query_suggester = Query_Suggestion(['qas.search.jsonl','qas.forum.jsonl'])

# dataset_path = 'collection.tsv'
# data = load_dataset(dataset_path)
# cleaned_texts = read_cleaned_texts('cleaned_texts.txt')
# tfidf_matrix, vectorizer = vectorize_texts(cleaned_texts)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster-search')
def cluster_index():
    return render_template('cluster-index.html')
    

@app.route('/load-dataset', methods=['POST'])
def load_dataset_route():
    global query_suggester,dataset, dataset_path, data, cleaned_texts, tfidf_matrix,  vectorizer,  query_refirment
    dataset = request.form['dataset']
   
    # Set the dataset paths based on the selected dataset
    if dataset == 'dataset1':
        print("The first dataset")

        with open("D:\DocumentVector.pkl", "rb") as file:
            tfidf_matrix = pickle.load(file)

        with open("D:\TfidfVector.pkl", "rb") as file:
            vectorizer = pickle.load(file)

    
        query_refirment=Query_Refinement()

         # Call the refine_queries_file method

        input_jsonl = r"D:\qas.search.jsonl"  
        output_csv = r"D:\refirment.csv"     
# 
        refirment =  query_refirment.refine_queries_file(input_jsonl, output_csv)
      
        query_suggester = Query_Suggestion(refirment, input_jsonl)

        dataset_path = r"D:\collection.tsv"

        data = load_dataset(dataset_path)
        read_data=r"D:\cleaned_texts.txt"
        cleaned_texts = read_cleaned_texts( read_data)
        
        # tfidf_matrix, vectorizer = vectorize_texts(cleaned_texts)
        print(" finesh")

    elif dataset == 'dataset2':
        print(" The secound dataset ")

        with open("D:\DocumentVector_clinical.pkl", "rb") as file:
            tfidf_matrix = pickle.load(file)

        with open("D:\TfidfVector_clinical.pkl", "rb") as file:
            vectorizer = pickle.load(file)

        query_refirment=Query_Refinement()
        
         # Call the refine_queries_file method
        input_jsonl = r"D:\queries.jsonl"  
        output_csv = r"D:\refirment2.csv"  
          
        refirment =  query_refirment.refine_queries_file(input_jsonl, output_csv)
      
        query_suggester = Query_Suggestion(refirment, input_jsonl)

        print("33333")
        dataset_path = r"D:\your_dataset.tsv"

        data = load_dataset2(dataset_path)

        print("55555")

        modified_data= r"D:\modified_dataset.csv"
        cleaned_texts2 =  pd.read_csv(modified_data)

        print("modified_data")
        # query_suggester = Query_Suggestion(['another.dataset1.jsonl','another.dataset2.jsonl'])
    
    
    return jsonify({"status": "Dataset loaded successfully"})

@app.route('/cluster-query', methods=['POST'])
def cluster_query():
    global dataset, tfidf_matrix, vectorizer

    query = request.form.get('query')
    dataset = request.form.get('dataset')
    print(f"Received query: {query}")
    print(f"Received dataset: {dataset}")
    processed_query = processtext(query)
    processed_query= processed_query['processed_text']
    print(processed_query)

    top_documents, cosine_similarities = get_documents_for_query(processed_query, tfidf_matrix, processor, vectorizer, data)
    documents = []
    if dataset == 'dataset1':
        for index, row in top_documents.iterrows():
            pid = row['pid']
            text = row['text']
            document = {'pid': pid, 'text': text}
            documents.append(document)
        
        return jsonify(top_documents=documents)

    elif dataset == 'dataset2':
        print("dataset2 ....data")
        for index, row in top_documents.iterrows():
            pid = row['doc_id']
            text = row['summary']  # Assuming you want the summary as text
            document = {'pid': pid, 'text': text}
            documents.append(document)
        print("yess.....")
        return jsonify(top_documents=documents)

   

@app.route('/suggest-query', methods=['GET'])
def suggest_query():
    print("sugg")
    global query_suggester ,query_refirment
    query = request.args.get('query', '')
    
    suggestions = query_suggester.suggest_similar_queries(query,n=5)
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True)
