from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import HTTPError
import re 

def crawlerThread(frontier, num_targets):
    visited = set()
    targets_found = []
    while frontier and len(targets_found) < num_targets:
        url = frontier.pop()
        visited.add(url)
        try:
            html = urlopen(url)
        except HTTPError as e:
            print(e)
            continue
        bs = BeautifulSoup(html, 'html.parser')

        # store page in mongo
        
        if bs.find("div", {"class":"fac-info"}) and bs.find("aside", {"aria-label":"faculty accolades"}):
            targets_found.append(url)
            print("FOUND", url)
        else:
            for a in bs.find_all('a', {"href":re.compile(r'^(\/|(https:\/\/www.cpp.edu)).*html')}, href = True):
                link = a.get('href')
                if not re.search("^https:\/\/www.cpp.edu", link):
                    link = "https://www.cpp.edu/" + link
                if re.search("^(https:\/\/www.cpp.edu).*html", link) and link not in visited:
                    frontier.add(link)
    print(targets_found)

seed = "https://www.cpp.edu/cba/international-business-marketing/index.shtml"
frontier = set()
frontier.add(seed)
crawlerThread(frontier, 10)