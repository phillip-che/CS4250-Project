from bs4 import BeautifulSoup
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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
    words = word_tokenize(text)
    #stopword removal
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    new_text = ' '.join(words)
    return new_text

