import json
import csv
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Query_Suggestion:
    def __init__(self, query_file, original_query_file):
        # تحميل الاستعلامات المنقحة من ملف CSV
        self.queries = []
        with open(query_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.queries.append(row[1])
        
        # إنشاء كائن TfidfVectorizer وتدريبه على الاستعلامات المنقحة
        self.vectorizer = TfidfVectorizer()
        self.query_vectors = self.vectorizer.fit_transform(self.queries)

        # تحميل الاستعلامات الأصلية من ملف JSONL
        self.original_queries = []
        self.query_ids = []
        with open(original_query_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                query = data["query"]
                query_id = data["qid"]
                self.original_queries.append(query)
                self.query_ids.append(query_id)

    def extract_question(self, query):
        # استخدام تعبير نمطي لاستخراج جزء السؤال من الاستعلام
        match = re.search(r'"query": "(.*)"', query)
        return match.group(1) if match else query

    def suggest_similar_queries(self, query, n=5):
        # تقسيم الاستعلام إلى كلمات
        query_tokens = nltk.word_tokenize(query)

        # حساب المتجه TF-IDF للاستعلام المدخل
        query_vector = self.vectorizer.transform([' '.join(query_tokens)])

        # حساب تشابه جيب التمام بين الاستعلام المدخل وكل استعلام منقح
        similarities = cosine_similarity(query_vector, self.query_vectors)

        # العثور على n من الاستعلامات الأكثر تشابهًا مع الاستعلام المدخل
        similar_indices = similarities.argsort()[0][-n-1:-1][::-1]
        similar_queries = [self.extract_question(self.original_queries[idx]) for idx in similar_indices]

        # إرجاع قائمة الأسئلة المشابهة
        return similar_queries 

# استخدام الكود
# query_file = 'refinement.csv'
# original_query_file = r"C:\Users\USER\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.search.jsonl"
# query_suggester = Query_Suggestion(query_file, original_query_file)
# suggestions = query_suggester.suggest_similar_queries("مثال على استعلام", n=5)
# print(suggestions)