from crawler import CalPolyCrawler
from index import Indexer
from parse import Parser
from ranker import Ranker
from query_transform import QueryTransformer


def build_index():
    crawler = CalPolyCrawler('https://www.cpp.edu/cba/international-business-marketing/index.shtml')
    targets_found = crawler.crawl(num_targets=22)

    parser = Parser()
    tokens_by_document, ordered_sentences_per_document, urls = parser.process_texts()
    
    indexer = Indexer()
    indexer.add_documents(ordered_sentences_per_document, urls)
    

build_index()

    