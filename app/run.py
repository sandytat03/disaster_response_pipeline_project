import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    
    #chart 1 data
    genre_names = list(genre_counts.index)
    genre_values = df.groupby('genre').count()['message']
    
    #chart 2 data
    disaster_responses = pd.DataFrame(df.drop(['id','message','original','genre'], axis=1).sum(axis=0)).reset_index()
    disaster_responses.rename(columns={'index':'disaster_response', 0:'count'}, inplace=True)
    disaster_responses.sort_values(by=['count'], ascending=False, inplace=True)
    disaster_categories = disaster_responses['disaster_response']
    disaster_categories_dis = disaster_responses['count']
    
    #chart 3 data
    disaster_responses_by_genre = df.drop(['id','message'], axis=1).groupby(['genre']).sum(axis=0).reset_index()
    y_direct = disaster_responses_by_genre.loc[0][1:].values
    y_news = disaster_responses_by_genre.loc[1][1:].values
    y_social = disaster_responses_by_genre.loc[2][1:].values
    
    #chart 4 data
    no_cat_per_sms = df.drop(['id'], axis=1).sum(axis=1)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )   
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=disaster_categories,
                    y=disaster_categories_dis,
                )   
            ],       
            'layout': {
                'title': 'Distribution of Disaster Categories',
                'yaxis': {
                    'title': "Count"
                }
            }
        },
        {
            'data': [
                Bar(
                    name='direct',
                    x=disaster_categories,
                    y=y_direct
                ),
                Bar(
                    name='news',
                    x=disaster_categories,
                    y=y_news
                ),
                Bar(
                    name='social',
                    x=disaster_categories,
                    y=y_social
                )
            ],
            'layout': {
                'title': 'Distribution of Disaster Categories by Genre',
                'yaxis': {
                    'title': "Count"
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=no_cat_per_sms,
                )   
            ],
            'layout': {
                'title': 'Number of Disaster Categories per Message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of disaster categories per message"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
