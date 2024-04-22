from bs4 import BeautifulSoup
import re
import string

STOP_WORDS = set(['a', 'an','and', 'the', 'is', 'are', 'in', 'on',
                  'at', 'to', 'for', 'with', 'of', 'as', 'by', 'that',
                  'this', 'it', 'be', 'or', 'not', 'from', 'you', 'your'])

def extract(soup):
    return soup.get_text(separator=' ')

def alter_text(text):
    #lowercase
    text = text.lower()
    #removing html tags and other characters 
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    #tokenizing
    words = text.split()
    #stopword removal
    words = [word for word in words if word not in STOP_WORDS]

    new_text = ' '.join(words)
    return new_text
