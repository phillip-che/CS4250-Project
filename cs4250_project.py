from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import HTTPError
from pymongo import MongoClient
import re 

def connectDB():
    DB_NAME = "CPP"
    DB_HOST = "localhost"
    DB_PORT = 27017

    try:
        client = MongoClient(host=DB_HOST, port=DB_PORT)
        db = client[DB_NAME]
        print("Database connected successfully")
        return db
    except:
        print("Database not connected successfully")

def crawlerThread(frontier, num_targets, pagesCol):
    visited = set()
    targets_found = []
    while frontier and len(targets_found) < num_targets:
        url = frontier.pop()
        visited.add(url)
        try:
            html = urlopen(url)
        except HTTPError as e:
            print(str(url) + " : " + str(e))
            continue
        bs = BeautifulSoup(html, 'html.parser')

        # doc = {"url": url, "html": str(bs)}
        # pagesCol.update(doc, doc, {"upsert": "true"})
        
        if bs.find("div", {"class":"fac-info"}) and bs.find("aside", {"aria-label":"faculty accolades"}):
            targets_found.append(url)
            print("FOUND", url)
        else:
            for a in bs.find_all('a', {"href":re.compile(r'^(\/|(https:\/\/www.cpp.edu)).*html')}, href = True):
                link = a.get('href').replace(" ", "")
                if not re.search("^https:\/\/www.cpp.edu", link):
                    link = "https://www.cpp.edu/" + link
                if re.search("^(https:\/\/www.cpp.edu).*html", link) and link not in visited:
                    frontier.add(link)
    print(targets_found)
    print(len(visited))    

def main():
    db = connectDB()
    pagesCol = db.pages
    seed = "https://www.cpp.edu/cba/international-business-marketing/index.shtml"
    frontier = set()
    frontier.add(seed)
    crawlerThread(frontier, 22, pagesCol)

if __name__ == "__main__":
    main()