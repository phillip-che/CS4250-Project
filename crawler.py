from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup
import re
from collections import deque
from urllib.parse import urljoin, urlparse
import pymongo

class CalPolyCrawler:
    def __init__(self, seed_url, db_name="CPPTESTexit", db_host="localhost", db_port=27017):
        self.seed_url = seed_url
        self.frontier = deque([seed_url])
        self.visited = set()
        self.client = pymongo.MongoClient(host=db_host, port=db_port)
        self.db = self.client[db_name]
        self.pages_col = self.db.pages
        self.faculty_list = []

    def is_target(self, soup):
        # Implement stopping logic here
        is_target = soup.find("div", {"class": "fac-info"}) and soup.find("aside", {"aria-label": "faculty accolades"})
        return is_target
    
    
    def safe_page(self, url, soup):
        doc = {"url": url, "html": str(soup)}
        self.pages_col.update_one({"url": url}, {'$set': doc}, upsert=True)


    def crawl(self):
        while self.frontier:
            url = self.frontier.popleft()
            self.visited.add(url)
            print("Crawling:", url)
            try:
                with urlopen(url) as response:
                    html_content = response.read()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    try:
                        self.safe_page(url, soup)
                    except Exception as e:
                        print(f'Error saving page {url} to MongoDB: {e}')
                    
                    if self.is_target(soup):
                        # some stopping logic
                        print('Tagret condition met.')
                        return

                    for link in soup.find_all('a', href=True):
                        abs_link = urljoin(url, link['href'].strip())
                        if urlparse(abs_link).netloc.endswith("cpp.edu") and abs_link not in self.visited:
                            self.frontier.append(abs_link)
                            self.visited.add(abs_link)
                            
            except (HTTPError, URLError) as e:
                print('Failed to access:', url)
                continue
            
        print('Target not found.')
        return None


if __name__ == '__main__':
    crawler = CalPolyCrawler('https://www.cpp.edu/cba/international-business-marketing/index.shtml')
    crawler.crawl()
