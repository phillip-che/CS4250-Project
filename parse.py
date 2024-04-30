from bs4 import BeautifulSoup
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
import re

class Parser:
    def __init__(self, db_name='CPP_PROJECT', db_host='localhost', db_port=27017):
        self.client = MongoClient(host=db_host, port=db_port)
        self.db = self.client[db_name]
        self.pages_col = self.db.pages
        self.vectorizer = CountVectorizer(lowercase=True, stop_words='english', token_pattern=r'\b[a-zA-Z]{2,}\b')
        self.lemmatizer = WordNetLemmatizer()

    # extract only faculty information from page
    def extract(self, soup):
        found = soup.find('div', {"class":"row pgtop"})
        return found.text if found else ""

    # prepare text for parsing
    def prepare_text(self, text):
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize spaces and remove unwanted backslashes
        text = " ".join(word for word in text.split() if not word.startswith("\\"))
        return text

    def process_text(self, text):
        text = self.prepare_text(text)
        
        # Tokenize text
        text_matrix = self.vectorizer.fit_transform([text])
        tokens = self.vectorizer.get_feature_names_out()

        # Map original text to tokens to preserve order
        ordered_tokens = [word for word in text.split() if word in tokens]

        # Lemmatize the ordered tokens
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in ordered_tokens]
        
        return lemmatized_tokens

if __name__ == "__main__":
    parser = Parser()

    # Test case
    html_content = parser.pages_col.find_one({"url":"https://www.cpp.edu/faculty/mcgood/index.shtml"})
    if html_content:
        soup = BeautifulSoup(html_content['html'], 'html.parser')
        words = parser.process_text(parser.extract(soup))
        print(words)
    else:
        print("No HTML content found for the URL")
