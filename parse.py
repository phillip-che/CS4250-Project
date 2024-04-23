from bs4 import BeautifulSoup
from pymongo import MongoClient
import re
import spacy

class Parser:
    def __init__(self, db_name='CPP_PROJECT', db_host='localhost', db_port=27017):
        self.client = MongoClient(host=db_host, port=db_port)
        self.db = self.client[db_name]
        self.pages_col = self.db.pages
        self.nlp = spacy.load('en_core_web_sm')

    # extract only faculty information of page
    def extract(self, soup):
        return soup.find('div', {"class":"row pgtop"}).text
    
    # prepare text for parsing
    def prepare_text(self, text):
        # lowercase
        text = text.lower()
        
        # removing html tags and other characters 
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\b\w{1}\b', '', text)
        text = " ".join(word for word in text.split() if not word.startswith("\\"))

        return text

    def process_text(self, text):

        text = self.prepare_text(text)
        
        # creating nlp doc object from text
        doc = self.nlp(text)
        
        # filtering, lemmatization, tokenization and punctuation removal
        words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

        return words

if __name__ == "__main__":
    parser = Parser()

    # test case
    html = parser.pages_col.find_one({"url":"https://www.cpp.edu/faculty/mcgood/index.shtml"})['html']
    soup = BeautifulSoup(html, 'html.parser')
    words = parser.process_text(parser.extract(soup))
    print(words)