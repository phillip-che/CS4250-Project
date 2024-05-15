# Cal Poly Pomona Business Search Engine

Cal Poly Pomona Business Search engine is a search engine system designed to crawl, parse, index, and retreive web pages concering the Cal Poly Pomona Business department. It utilizes MongoDB to store documents, index, and raw pages and displays results using a Flask server (`app.py`) to get queries and display results. It used a robust ranking system to determine which documents to display to the user, looking at both domain-specific and general semantics.


## Features

- **Web Crawling**: Automatically discover and retrieve web pages.
- **Content Parsing**: Extract relevant content from raw web pages and train models.
- **Indexing**: Efficiently index the content to facilitate quick searches.
- **Query Processing**: Transform queries to improve search results.
- **Ranking**: Rank results based on relevance to the query.
- **Web Interface**: A simple and intuitive Flask-based web interface for performing searches.

## Installation

1. **Clone the repository**
2. **Ensure all dependencies are installed**
3. **Build the MongoDB index using `main.py`**
4. **Start Flask server `app.py` and run queries**

