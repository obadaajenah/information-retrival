import json
import csv
def query_json(filename ):

# اقرأ محتوى ملف CSV
    with open(filename, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        queries = []
        for row in csvreader:
            query_id = row.pop('query_id')
            combined_query = ' '.join(value for key, value in row.items() if value)
            queries.append({"qid": query_id, "query": combined_query})

    # احفظ كملف JSONL
    with open('D:\queries.jsonl', 'w', encoding='utf-8') as jsonlfile:
        for query in queries:
            jsonlfile.write(json.dumps(query) + '\n')


filename='D:\queries.csv'
query_json(filename)