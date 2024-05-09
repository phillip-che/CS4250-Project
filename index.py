from pymongo import MongoClient
from collections import defaultdict
import numpy as np
import math
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

class Indexer:
    def __init__(self, db_name='CPP_PROJECT', db_host='localhost', db_port=27017):
        self.client = MongoClient(host=db_host, port=db_port)
        self.db = self.client[db_name]
        self.documents_collection = self.db.documents
        self.index_collection = self.db.index
        self.doc_count = self.documents_collection.count_documents({})
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.word2vec_model = Word2Vec.load('./models/word2vec.model')
    
    def add_documents(self, documents):
        for doc_id, text in enumerate(documents):
            self.add_document(text, doc_id)

    def add_document(self, text, doc_id):
        text_str = " ".join(text)  

        word_vectors = [self.word2vec_model.wv[word] for word in text if word in self.word2vec_model.wv]
        if word_vectors:
            aggregated_vector = np.mean(word_vectors, axis=0)
        else:
            aggregated_vector = np.zeros(self.word2vec_model.vector_size)
            
        word_vector_pretrained = self.model.encode([text_str])[0]

        term_frequency = self.get_term_frequency(text)
        tf_idf = self.calculate_tf_idf(term_frequency)
        self.documents_collection.update_one(
            {'_id': doc_id}, 
            {
                '$set': {
                    '_id': doc_id,
                    'text': text_str,
                    'word_vector_pretrained': word_vector_pretrained.tolist(), 
                    'word_vector_word2vec': aggregated_vector.tolist(), 
                    'tf_idf': dict(tf_idf)  
                }
            }, 
            upsert=True
        )

        self.update_inverted_index(term_frequency, doc_id)
        self.doc_count += 1  

    def update_inverted_index(self, term_frequency, doc_id):
        for token in term_frequency:
            self.index_collection.update_one(
                {'token': token},
                {'$inc': {'doc_freq': 1}, '$addToSet': {'documents': doc_id}},
                upsert=True
            )

    def get_term_frequency(self, text):
        term_frequency = defaultdict(int)
        for token in text:
            term_frequency[token] += 1
        return term_frequency

    def calculate_tf_idf(self, term_frequency):
        tf_idf = defaultdict(float)
        for token, tf in term_frequency.items():
            result = self.index_collection.find_one({'token': token}, {'doc_freq': 1})
            doc_freq = result['doc_freq'] if result else 0
            idf = math.log((self.doc_count + 1) / (doc_freq + 1)) if doc_freq else 0 
            tf_idf[token] = tf * idf
        return tf_idf

    def query_index(self, query_text):
        # THE QUERY TEXT TRANSFORMATION WILL OCCUR ELSEWHERE JUST FOR TESTING
        query_tokens = query_text.split()

        query_term_frequency = self.get_term_frequency(query_tokens)

        query_vector = self.model.encode([query_text])[0]

        results = []
        for doc in self.documents_collection.find():
            doc_vector = np.array(doc['word_vector_pretrained'])
            doc_similarity = np.dot(doc_vector, query_vector) / (np.linalg.norm(doc_vector) * np.linalg.norm(query_vector))
            
            tf_idf_similarity = sum(doc['tf_idf'].get(token, 0) * self.calculate_tf_idf({token: query_term_frequency[token]}).get(token, 0) for token in query_tokens)
            
            combined_similarity = 0.5 * doc_similarity + 0.5 * tf_idf_similarity
            results.append((doc['_id'], combined_similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def update_all_tfidf(self):
        documents = list(self.documents_collection.find({}, {'_id': 1, 'text': 1}))
        doc_ids = [doc['_id'] for doc in documents]
        texts = [doc['text'] for doc in documents]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)

        for idx, doc_id in enumerate(doc_ids):
            tfidf_vector = tfidf_matrix[idx].todense().tolist()[0]
            tfidf_dict = {term: tfidf_vector[col] for term, col in vectorizer.vocabulary_.items()}
            
            self.documents_collection.update_one(
                {'_id': doc_id},
                {'$set': {'tf_idf': tfidf_dict}}
            )


if __name__ == "__main__":
    indexer = Indexer()
    indexer.add_document(['new', 'document', 'new', 'brand', 'hello', 'for', 'test', 'just', 'just'], 'doc9')
    indexer.add_document(['newest', 'documents', 'document'], 'doc8')
    indexer.add_document(['greetings', 'earth', 'earth', 'new'], 'doc10')
    indexer.update_all_tfidf() 
    query_text = "hello world"
    print(indexer.query_index(query_text))
