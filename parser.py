from bs4 import BeautifulSoup
from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import spacy

class Parser:
    def __init__(self, db_name='CPP_PROJECT', db_host='localhost', db_port=27017):
        self.client = MongoClient(host=db_host, port=db_port)
        self.db = self.client[db_name]
        self.pages_col = self.db.pages

    # extract only faculty information of page
    def extract(self, soup):
        return soup.find('div', {"class":"row pgtop"}).text

    def alter_text(self, text):
        stop_words = set(stopwords.words('english'))
        sp = spacy.load('en_core_web_sm')
        lemmatizer = WordNetLemmatizer()

        # lowercase
        text = text.lower()
        # removing html tags and other characters 
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\b\w{1}\b', '', text)
        text = " ".join(word for word in text.split() if not word.startswith("\\"))
        
        # tokenizing
        doc = sp(text)
        # stopword removal
        words = set(lemmatizer.lemmatize(token.text) for token in doc if token.text not in stop_words)

        return words


if __name__ == "__main__":
    parser = Parser()

    # test case
    html = parser.pages_col.find_one({"url":"https://www.cpp.edu/faculty/mcgood/index.shtml"})['html']
    soup = BeautifulSoup(html, 'html.parser')
    words = parser.alter_text(parser.extract(soup))
    print(words)