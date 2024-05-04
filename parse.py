from bs4 import BeautifulSoup
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
import re
import joblib

class CustomTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def tokenize_and_lemmatize(self, text):
        tokens = nltk.word_tokenize(text)
        return [self.lemmatizer.lemmatize(token) for token in tokens]

class Parser:
    def __init__(self, db_name='CPPTESTexit', db_host='localhost', db_port=27017):
        self.client = MongoClient(host=db_host, port=db_port)
        self.db = self.client[db_name]
        self.pages_col = self.db.pages
        custom_tokenizer = CustomTokenizer()
        self.vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 5), tokenizer=custom_tokenizer.tokenize_and_lemmatize)

    def extract(self, soup):
        found = soup.find('div', {"class": "row pgtop"})
        return found.text if found else ""

    def prepare_text(self, text):
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = " ".join(word for word in text.split() if not word.startswith("\\"))
        return text

    def process_texts(self):
        documents = []
        doc_ids = []
        # Process all faculty pages
        for html_content in self.pages_col.find({"url": {"$regex": "^https://www.cpp.edu/faculty/"}}):
            if html_content:
                doc_ids.append(html_content['_id'])
                soup = BeautifulSoup(html_content['html'], 'html.parser')
                extracted_text = self.extract(soup)
                prepared_text = self.prepare_text(extracted_text)
                documents.append(prepared_text)
        
        # Fit the vectorizer on the entire corpus (Gives us the vocabulary)
        self.vectorizer.fit(documents)
        joblib.dump(self.vectorizer, './models/vectorizer.pkl')

        # Transform each document and collect tokens and counts (Keeps the order of the documents)
        document_tokens = {}
        for doc_id, document in zip(doc_ids, documents):
            vector = self.vectorizer.transform([document]).toarray()
            feature_names = self.vectorizer.get_feature_names_out()
            document_tokens[doc_id] = dict(zip(feature_names, vector.flatten()))

        return document_tokens

if __name__ == "__main__":
    parser = Parser()
    tokens_by_document = parser.process_texts()
    for doc_id, tokens in tokens_by_document.items():
        print(f"Document ID: {doc_id}")
        for token, count in tokens.items():
            if count > 0:
                print(f"{token}: {count}")