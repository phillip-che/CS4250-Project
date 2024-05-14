from flask import Flask, request, render_template
from ranker import Ranker
from query_transform import QueryTransformer

app = Flask(__name__)

def transform_query(query_text):
    transformer = QueryTransformer()
    transformed_query = transformer.transform_query(query_text)
    return transformed_query


# build_index()
    
def get_relevant_documents(query_text):
    transformer = QueryTransformer()
    transformed_query = transformer.transform_query(query_text)
    
    ranker = Ranker()
    tokens, word2vec_vector, sentence_embeddings, tfidf_dict = transformed_query
    results = ranker.get_similar_docs(word2vec_vector, tfidf_dict, sentence_embeddings, tokens)
    return results[:5]

@app.route('/', methods=['GET'])
def index():
    query = request.args.get('query')
    if query is None:
        return render_template('index.html')
    
    relevant_docs = get_relevant_documents(query)

    return render_template('index.html', relevant_docs=relevant_docs)

if __name__ == '__main__':
    app.run(debug=True)
