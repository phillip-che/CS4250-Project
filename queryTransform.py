import re
import spacy

class QueryTransformer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(spacy.lang.en.stop_words.STOP_WORDS)

    def transform_query(self, query_text):
        # lowercase
        query_text = query_text.lower()
        
        # removing html tags and other characters 
        query_text = re.sub(r'<[^>]+>', '', query_text)
        query_text = re.sub(r'\b\w{1}\b', '', query_text)
        query_text = " ".join(word for word in query_text.split() if not word.startswith("\\"))

        return self.lemmatize_tokens(query_text)

    def lemmatize_tokens(self, text):
        # creating nlp doc object from text
        doc = self.nlp(text)
        
        # lemmatization, tokenization, stop word removal, and punctuation removal
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

        return tokens

if __name__ == "__main__":
    transformer = QueryTransformer()
    query_text = "I work at McDonalds and try my hardest to create the greatest French Fries. Test query text!"
    transformed_query = transformer.transform_query(query_text)
    print(transformed_query)