from sentence_transformers import SentenceTransformer
import spacy
from pymongo import MongoClient
from nltk.util import ngrams

class Ranker:
    def __init__(self, db_name='CPP_PROJECT', db_host='localhost', db_port=27017):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = MongoClient(host=db_host, port=db_port)
        self.db = self.client[db_name]
        self.documents_collection = self.db.documents
        self.index_collection = self.db.index
        self.nlp = spacy.load('en_core_web_sm')
        
    # METHOD TO RANK RELEVANT DOCS ACCORDING TO QUERY TERMS
    def rank_relevant_docs(self, query, doc_ids):
        pass
    
    # METHOD TO FIND RELEVANT DOCS ACCORDING TO QUERY TERMS
    def get_relevant_docs(self, query):
        pass
    
    def get_entities(self, query):
        doc = self.nlp(query)
        return [ent.text for ent in doc.ents]
    
    def get_grams(self, query):
        bigram = list(ngrams(query, 2))
        trigram = list(ngrams(query, 3))
        return {'bigram': bigram, 'trigram': trigram}
    
    # IMPLEMENTATION WILL BE DIFFERENT FROM INDEX
    def calculate_tf_idf(self, query):
        pass
    
    
    
    
    
    