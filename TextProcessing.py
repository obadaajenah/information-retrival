from fastapi import FastAPI, Request
import nltk
from nltk.tokenize import word_tokenize
import inflect
import re
from num2words import num2words
import uvicorn

class TextProcessor:
    
    def __init__(self):
        self.stemmer = nltk.PorterStemmer()
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.inflect_engine = inflect.engine()

    @staticmethod
    def number_to_words(text):
        words = word_tokenize(text)
        converted_words = []
        for word in words:
            if word.isdigit():
                try:
                    if int(word) > 999999999:
                        converted_words.append("[Number Too Large]")
                    else:
                        converted_word = inflect.engine().number_to_words(word)
                        converted_words.append(converted_word)
                except Exception as e:
                    print(f"Error converting number to words: {e}")
                    continue
            else:
                converted_words.append(word)
        return ' '.join(converted_words)

    @staticmethod
    def cleaned_text(text):
        try:
            cleaned_text = re.sub(r'\\n', ' ', text)
            cleaned_text = re.sub(r'\W', ' ', cleaned_text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            return cleaned_text
        except Exception as e:
            print(f"Error cleaning text: {e}")
            raise e

    @staticmethod
    def normalization_example(text):
        return text.lower()

    @staticmethod
    def stemming_example(text):
        try:
            words = word_tokenize(text)
            stemmed_words = [nltk.PorterStemmer().stem(word) for word in words]
            return ' '.join(stemmed_words)
        except Exception as e:
            print(f"Error stemming text: {e}")
            raise e

    @staticmethod
    def lemmatization_example(text):
        try:
            words = word_tokenize(text)
            lemmatized_words = [nltk.WordNetLemmatizer().lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
        except Exception as e:
            print(f"Error lemmatizing text: {e}")
            raise e

    @staticmethod
    def remove_stopwords(text):
        try:
            stop_words = set(nltk.corpus.stopwords.words('english'))
            words = word_tokenize(text)
            filtered_words = [word for word in words if word.lower() not in stop_words]
            return ' '.join(filtered_words)
        except Exception as e:
            print(f"Error removing stopwords: {e}")
            raise e

    @staticmethod
    def remove_punctuation(text):
        try:
            return re.sub(r'[^\w\s,]', '', text)
        except Exception as e:
            print(f"Error removing punctuation: {e}")
            raise e
        
def processtext(text):
    
        processed_text = TextProcessor.cleaned_text(text)
        processed_text = TextProcessor.normalization_example(processed_text)
        processed_text = TextProcessor.stemming_example(processed_text)
        processed_text = TextProcessor.lemmatization_example(processed_text)
        processed_text = TextProcessor.remove_stopwords(processed_text)
        processed_text = TextProcessor.number_to_words(processed_text)
        processed_text = TextProcessor.remove_punctuation(processed_text)
        return {"processed_text": processed_text}
      
    
app = FastAPI()

@app.post("/process_text")
async def process_text(request: Request):
    try:
        data = await request.json()
        text = data.get("text")
     
        processed_text = TextProcessor.cleaned_text(text)
        processed_text = TextProcessor.normalization_example(processed_text)
        processed_text = TextProcessor.stemming_example(processed_text)
        processed_text = TextProcessor.lemmatization_example(processed_text)
        processed_text = TextProcessor.remove_stopwords(processed_text)
        processed_text = TextProcessor.number_to_words(processed_text)
        processed_text = TextProcessor.remove_punctuation(processed_text)
        return {"processed_text": processed_text}
    except Exception as e:
        print(f"Error processing text: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
