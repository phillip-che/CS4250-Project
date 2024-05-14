from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import numpy as np
import joblib

class Ranker:
    def __init__(self, db_name='CPP_PROJECT', db_host='localhost', db_port=27017, tfidf_vectorizer_path='./models/tfidf_vectorizer.pkl'):
        self.client = MongoClient(host=db_host, port=db_port)
        self.db = self.client[db_name]
        self.documents_collection = self.db.documents
        self.index_collection = self.db.index
        self.tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.feature_name_dict = {name: idx for idx, name in enumerate(self.feature_names)}

    def get_doc_tokens(self, doc_id):
        tokens = set()
        for entry in self.index_collection.find({'documents': doc_id}):
            tokens.add(entry['token'])
        return tokens

    def get_similar_docs(self, word2vec_vector, tfidf_dict, sentence_embeddings, tokens):
        docs = []

        # Create a full-length TF-IDF vector for the query
        query_tfidf_vector = np.zeros(len(self.feature_names))
        for term, value in tfidf_dict.items():
            if term in self.feature_name_dict:
                query_tfidf_vector[self.feature_name_dict[term]] = value

        for doc in self.documents_collection.find():
            doc_word_vector_word2vec = doc.get('word_vector_word2vec')
            doc_word_vector_pretrained = doc.get('word_vector_pretrained')
            
            word2vec_similarity = 0
            tfidf_similarity = 0
            sentence_similarity = 0
            
            if word2vec_vector:
                word2vec_similarity = cosine_similarity([word2vec_vector], [np.array(doc_word_vector_word2vec)]) if doc_word_vector_word2vec is not None else np.array([[0]])
            if query_tfidf_vector.any():
                tfidf_similarity = cosine_similarity([query_tfidf_vector], [self._get_doc_tfidf_vector(doc)])
            if sentence_embeddings.any():
                sentence_similarity = cosine_similarity(sentence_embeddings, [np.array(doc_word_vector_pretrained)]) if doc_word_vector_pretrained is not None else np.array([[0]])
            

            # Token overlap score (Jaccard similarity)
            doc_tokens = self.get_doc_tokens(doc['_id'])
            query_tokens = set(tokens)
            token_overlap_score = len(doc_tokens.intersection(query_tokens)) / len(doc_tokens.union(query_tokens)) if doc_tokens or query_tokens else 0

            print(f"Token overlap score: {token_overlap_score}")

            # Calculate the final similarity score
            similarity_score = (word2vec_similarity + tfidf_similarity + sentence_similarity + token_overlap_score) / 4
            print(f"Similarity score: {similarity_score}")
            
            text = doc['text']
            words = text.split(' ')
            try:
                email_index = words.index('Email')
            except ValueError:
                email_index = 20
                
            doc_snippet = " ".join(words[:email_index])

            docs.append((doc['url'], similarity_score[0][0], doc_snippet))

        return sorted(docs, key=lambda x: x[1], reverse=True)

    def _get_doc_tfidf_vector(self, doc):
        doc_tfidf_vector = np.zeros(len(self.feature_names))
        for term, value in doc['tf_idf'].items():
            if term in self.feature_name_dict:
                doc_tfidf_vector[self.feature_name_dict[term]] = value
        return doc_tfidf_vector

    def get_relevant_docs_by_tokens(self, tokens):
        doc_ids = set()
        for token in tokens:
            result = self.index_collection.find_one({'token': token})
            if result:
                doc_ids.update(result['documents'])
        return list(doc_ids)
