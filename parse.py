from bs4 import BeautifulSoup
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import nltk
import re
import joblib

class CustomTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def tokenize_and_lemmatize(self, text):
        tokens = nltk.word_tokenize(text)
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def tokenize_and_lemmatize_word2vec(self, text):
        sentences = nltk.sent_tokenize(text)
        return [self.lemmatizer.lemmatize(token) for sentence in sentences for token in nltk.word_tokenize(sentence)]

class Parser:
    def __init__(self, db_name='CPP_PROJECT', db_host='localhost', db_port=27017):
        self.client = MongoClient(host=db_host, port=db_port)
        self.db = self.client[db_name]
        self.pages_col = self.db.pages
        self.custom_tokenizer = CustomTokenizer()
        self.vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 5), tokenizer=self.custom_tokenizer.tokenize_and_lemmatize)

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
        ordered_tokens_per_document = []

        # Process all faculty pages
        for html_content in self.pages_col.find({"url": {"$regex": "^https://www.cpp.edu/faculty/"}}):
            if html_content:
                doc_ids.append(html_content['_id'])
                soup = BeautifulSoup(html_content['html'], 'html.parser')
                extracted_text = self.extract(soup)
                prepared_text = self.prepare_text(extracted_text)
                documents.append(prepared_text)
                ordered_tokens_per_document.append(self.custom_tokenizer.tokenize_and_lemmatize_word2vec(prepared_text))

        # Fit the vectorizer on the entire corpus (Gives us the vocabulary)
        self.vectorizer.fit(documents)
        joblib.dump(self.vectorizer, './models/vectorizer.pkl')
        
        # Fit the word2vec model on the combined tokens from all documents
        word2vec_model = Word2Vec(sentences=ordered_tokens_per_document, vector_size=100, window=5, min_count=1, workers=4)
        word2vec_model.save('./models/word2vec.model')

        # Transform each document and collect tokens and counts (Keeps the order of the documents)
        document_tokens = {}
        for doc_id, document in zip(doc_ids, documents):
            vector = self.vectorizer.transform([document]).toarray()
            feature_names = self.vectorizer.get_feature_names_out()
            document_tokens[doc_id] = dict(zip(feature_names, vector.flatten()))

        # Document tokens is a dictionary where the key is the document id and the value is a dictionary of tokens and their counts
        # Ordered tokens per document is a list of lists where each list is the tokens of a document
        return document_tokens, ordered_tokens_per_document

if __name__ == "__main__":
    parser = Parser()
    tokens_by_document, ordered_tokens_per_document = parser.process_texts()
    #print(tokens_by_document[list(tokens_by_document.keys())[0]])  # Print tokens of the first document
    print(ordered_tokens_per_document[0])  
