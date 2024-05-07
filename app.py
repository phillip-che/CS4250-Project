from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    query = request.args.get('query')
    if query is None:
        return render_template('index.html')
    
    relevant_docs = get_relevant_docs(query)
    return render_template('index.html', relevant_docs=relevant_docs)

def get_relevant_docs(query):
    return [
        ('http://google.com', 'Google is a search engine'),
        ('http://facebook.com', 'Facebook is a social network')
    ]

if __name__ == '__main__':
    app.run(debug=True)
