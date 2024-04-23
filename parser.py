from bs4 import BeautifulSoup
import re
import string
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

sp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def extract(soup):
    return soup.get_text(separator=' ')

def alter_text(text):
    #lowercase
    text = text.lower()
    #removing html tags and other characters 
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    #tokenizing
    doc = sp(text)
    #stopword removal
    words = [lemmatizer.lemmatize(token.text) for token in doc if token.text not in stop_words]

    new_text = ' '.join(words)
    return new_text

