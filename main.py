from crawler import CalPolyCrawler
from index import Indexer
from parse import Parser


def build_index():
    crawler = CalPolyCrawler('https://www.cpp.edu/cba/international-business-marketing/index.shtml')
    targets_found = crawler.crawl(num_targets=22)

    parser = Parser()
    tokens_by_document, ordered_sentences_per_document = parser.process_texts()
    
    indexer = Indexer()
    indexer.add_documents(ordered_sentences_per_document)
    

build_index()
    
    