from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup
import re
from collections import deque
from urllib.parse import urljoin, urlparse
import pymongo

class CalPolyCrawler:
    def __init__(self, seed_url, db_name="CPP_PROJECT", db_host="localhost", db_port=27017):
        self.seed_url = seed_url
        self.frontier = deque([seed_url])
        self.visited = set()
        self.client = pymongo.MongoClient(host=db_host, port=db_port)
        self.db = self.client[db_name]
        self.pages_col = self.db.pages
        self.faculty_list = []

    def is_target(self, soup):
        # Implement stopping logic here
        return soup.find("div", {"class": "fac-info"}) and soup.find("aside", {"aria-label": "faculty accolades"})
    
    def save_page(self, url, soup):
        doc = {"url": url, "html": str(soup)}
        self.pages_col.update_one({"url": url}, {'$set': doc}, upsert=True)

    def crawl(self, num_targets):
        targets_found = []
        while self.frontier and len(targets_found) < num_targets:
            url = self.frontier.popleft()
            print("Crawling:", url)
            try:
                with urlopen(url) as response:
                    html_content = response.read()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    try:
                        self.save_page(url, soup)
                    except Exception as e:
                        print(f'Error saving page {url} to MongoDB: {e}')
                    
                    if self.is_target(soup):
                        targets_found.append(url)
                        continue

                    for link in soup.find_all('a', href=True):
                        abs_link = urljoin(url, link['href'].strip())
                        if urlparse(abs_link).netloc.endswith("cpp.edu") and abs_link not in self.visited:
                            self.frontier.append(abs_link)
                            self.visited.add(abs_link)
                            
            except (HTTPError, URLError) as e:
                print('Failed to access:', url)
                continue
        
        return targets_found

