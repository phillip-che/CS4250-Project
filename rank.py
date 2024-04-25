from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
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
        #retrieve the relevant docs from the query and calculate the tf-idf
        relevant_docs = self.get_relevant_docs(query)
        query_tf_idf = self.calculate_tf_idf(query)

        #generate sentence embedding (vector)
        query_embedding = self.model.encode([query])[0]
        results = []

        for doc_id, doc_text in relevant_docs:
          # Calculate tf-idf of the current doc and generate vector 
          doc_tf_idf = self.calculate_tf_idf(doc_text)
          doc_embedding = self.model.encode([doc_text])[0]

          #calculate cosine similarity beween query and document tf-idf, same goes for embedding
          tf_idf_similarity = self.calculate_cosine_similarity(query_tf_idf, doc_tf_idf)
          embedding_similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]

          #combine the embedding and tf-idf similarity using 50% each
          combined_similarity = 0.5 * tf_idf_similarity + 0.5 * embedding_similarity
          results.append((doc_id, combined_similarity))
       results.sort(key=lambda x:x[1], reverse=True)
       return results
    
    # METHOD TO FIND RELEVANT DOCS ACCORDING TO QUERY TERMS
    def get_relevant_docs(self, query):
        entities = self.get_entities(query)
        grams = self.get_grams(query)
        relevant_doc_ids = set()

        #search through relevant docs based on entities
        for entity in entities:
          result = self.index_collection.find_one({'token': entity})
          if result:
            relevant_doc_ids.update(result['documents'])

        #search through relevant docs based on bigrams and trigrams
        for gram_type in ['bigram', 'trigram']:
          for gram in grams[gram_type]:
            gram_str = ' '.join(gram)
            result = self.index_collection.find_one({'token': gram_str})
            if result:
              relevant_doc_ids.update(result['documents'])
        relevant_docs = []

        #retrieve relevant doc IDs with the text
        for doc_id in relevant_doc_ids:
          doc = self.documents.collection.find_one({'_id': doc_id})
          id doc:
            relevant_docs.append((doc_id, doc['text']))
        return relevant_docs
    
    def get_entities(self, query):
        doc = self.nlp(query)
        return [ent.text for ent in doc.ents]
    
    def get_grams(self, query):
        words = query.split()
        bigram = list(ngrams(words, 2))
        trigram = list(ngrams(words, 3))
        return {'bigram': bigram, 'trigram': trigram}
    
    # IMPLEMENTATION WILL BE DIFFERENT FROM INDEX
    def calculate_tf_idf(self, query):
        #fit and transorm the vectorizer
        tfidf_vectorizer = TfidfVectorizer()
        query_tfidf_matrix = tfidf_vectorizer.fit_transform([query])
        #get term names from the vectorizer
        query_names = tfidf_vectorizer.get_feature_names_out()
        #for storing tf-idf scores
        query_tfidf = {}
        #go over the columns of the tf-idf matrix
        for col in query_tfidf_matrix.nonzero()[1]:
          query_tfidf[feature_names[col]] = query_tfidf_matrix[0, col]
        return query_tfidf

    def calculate_cosine_similarity(self, vector1, vector2):
      return cosine_similarity([list(vector1.values())], [list(vector2.values())])[0][0]
    
    
    
    
    
    
