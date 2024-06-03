
import sys
import pandas as pd
import pickle

def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path, delimiter='\t', header=None, names=['pid', 'text'])
    except pd.errors.ParserError as e:
        print(f"Error reading the dataset file: {e}")
        sys.exit(1)
    return data

def read_cleaned_texts(file_path):
    try:
        with open(file_path, 'r' , encoding='utf-8') as file:
            lines = file.readlines()
            return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        sys.exit(1)


def save_vectors_to_binary_file(vector, filename):
        
        with open(filename, "wb") as file:
          pickle.dump(vector, file)

def convert_csv_to_tsv(input_csv_path, output_tsv_path):
    try:
        # قراءة ملف CSV
        df = pd.read_csv(input_csv_path)
        
        # حفظ البيانات في ملف TSV
        df.to_csv(output_tsv_path, sep='\t', index=False)
        print(f"تم تحويل الملف بنجاح وحفظه في {output_tsv_path}")
    except Exception as e:
        print(f"حدث خطأ أثناء تحويل الملف: {e}")

# مثال على كيفية استخدام التابع
input_csv_path = r"D:\your_dataset.csv"
output_tsv_path = r"D:\your_dataset.tsv"
convert_csv_to_tsv(input_csv_path, output_tsv_path)

def load_dataset2(file_path):
    try:
        data = pd.read_csv(file_path, delimiter='\t')
    except pd.errors.ParserError as e:
        print(f"Error reading the dataset file: {e}")
        sys.exit(1)
    return data

def read_csv_to_array(file_path):
    # قراءة ملف CSV باستخدام pandas
    df = pd.read_csv(file_path)
    
    # تحويل إطار البيانات إلى مصفوفة من القوائم
    data_array = df.values.tolist()
    
    return data_array



# def load_vectors_from_binary_file(filename="data_vectors.pkl"):
#     # قراءة المتجهات من ملف ثنائي باستخدام pickle
#     with open(filename, 'rb') as file:
#         return pickle.load(file)


# # تعريف تابع لحفظ مصفوفة التشابه في ملف
# def save_similarity_matrix(similarity_matrix, file_path):
#     # حفظ مصفوفة التشابه في الملف
#     np.save(file_path, similarity_matrix)
#     print("Similarity Matrix saved successfully!")

# # تعريف تابع لتحميل مصفوفة التشابه من الملف
# def load_similarity_matrix(file_path):
#     # قم بتحميل مصفوفة التشابه من الملف
#     similarity_matrix = np.load(file_path)
#     return similarity_matrix

