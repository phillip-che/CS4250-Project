import re
import joblib
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

class CustomTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def tokenize_and_lemmatize(self, text):
        tokens = nltk.word_tokenize(text)
        return [self.lemmatizer.lemmatize(token) for token in tokens]

class QueryTransformer:
    def __init__(self, vectorizer_path='./models/vectorizer.pkl', tfidf_vectorizer_path='./models/tfidf_vectorizer.pkl'):
        self.vectorizer = joblib.load(vectorizer_path)
        self.tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        self.word2vec_model = Word2Vec.load('./models/word2vec.model')
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def transform_query(self, query_text):
        # Lowercase conversion and removing HTML tags
        query_text = query_text.lower()
        query_text = re.sub(r'<[^>]+>', '', query_text)
        
        # Remove punctuation
        query_text = re.sub(r'[^\w\s]', '', query_text)
        
        # Removing single-character words and handling backslashes
        query_text = " ".join(word for word in query_text.split() if not word.startswith("\\"))

        sentence_embeddings = self.model.encode([query_text])
        
        # Transform the cleaned text using the pre-loaded vectorizer
        transformed_vector = self.vectorizer.transform([query_text])
        feature_names = self.vectorizer.get_feature_names_out()

        # Identify non-zero elements in the transformed vector to get tokens present in the query
        non_zero_indices = transformed_vector.nonzero()[1]
        tokens_present = [feature_names[index] for index in non_zero_indices]
        
        word_embeddings = [self.word2vec_model.wv[word] for word in tokens_present if word in self.word2vec_model.wv]
        if word_embeddings:
            aggregated_vector = sum(word_embeddings) / len(word_embeddings)
        else:
            aggregated_vector = None

        # Calculate TF-IDF values for the query
        tfidf_vector = self.tfidf_vectorizer.transform([query_text]).todense().tolist()[0]
        tfidf_dict = {term: tfidf_vector[col] for term, col in self.tfidf_vectorizer.vocabulary_.items() if tfidf_vector[col] > 0}

        return tokens_present, aggregated_vector, sentence_embeddings, tfidf_dict

