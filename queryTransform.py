import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

class QueryTransformer:
    def __init__(self):
        self.vectorizer = CountVectorizer(lowercase=True, stop_words='english', token_pattern=r'\b[a-zA-Z]{2,}\b')
        self.lemmatizer = WordNetLemmatizer()

    def transform_query(self, query_text):
        # Lowercase conversion and removing HTML tags
        query_text = query_text.lower()
        query_text = re.sub(r'<[^>]+>', '', query_text)
        
        # Remove punctuation
        query_text = re.sub(r'[^\w\s]', '', query_text)
        
        # Removing single-character words and handling backslashes
        query_text = " ".join(word for word in query_text.split() if not word.startswith("\\"))

        # Tokenize the cleaned text
        self.vectorizer.fit([query_text])
        tokens = self.vectorizer.get_feature_names_out()

        # Map original text to tokens to preserve order
        ordered_tokens = [token for token in query_text.split() if token in tokens]

        # Lemmatize the filtered tokens
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in ordered_tokens]

        return lemmatized_tokens

if __name__ == "__main__":
    transformer = QueryTransformer()
    query_text = "I work at McDonalds and try my hardest to create the greatest French Fries. Test query text!"
    transformed_query = transformer.transform_query(query_text)
    print(transformed_query)
