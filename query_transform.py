import re
import joblib
from nltk.stem import WordNetLemmatizer
import nltk

class CustomTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def tokenize_and_lemmatize(self, text):
        tokens = nltk.word_tokenize(text)
        return [self.lemmatizer.lemmatize(token) for token in tokens]

class QueryTransformer:
    def __init__(self, vectorizer_path='fitted_vectorizer.pkl'):
        self.vectorizer = joblib.load(vectorizer_path)

    def transform_query(self, query_text):
        # Lowercase conversion and removing HTML tags
        query_text = query_text.lower()
        query_text = re.sub(r'<[^>]+>', '', query_text)
        
        # Remove punctuation
        query_text = re.sub(r'[^\w\s]', '', query_text)
        
        # Removing single-character words and handling backslashes
        query_text = " ".join(word for word in query_text.split() if not word.startswith("\\"))

        # Transform the cleaned text using the pre-loaded vectorizer
        transformed_vector = self.vectorizer.transform([query_text])
        feature_names = self.vectorizer.get_feature_names_out()

        # Identify non-zero elements in the transformed vector to get tokens present in the query
        non_zero_indices = transformed_vector.nonzero()[1]
        tokens_present = [feature_names[index] for index in non_zero_indices]

        return tokens_present

if __name__ == "__main__":
    transformer = QueryTransformer()
    query_text = "I work at McDonalds and try my hardest to create the greatest French Fries. Test query text!"
    transformed_query = transformer.transform_query(query_text)
    print(transformed_query)