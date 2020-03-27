# 1. IMPORT LIBRARIES

import sys

import sqlite3
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import pickle

#2. LOAD DATA FROM DATABASE

def load_data(database_filepath):
    """Load data: load data from SQLite database file - output from process_data.py

    Parameters:
    database_filepath (SQLite database .db): SQLite database filepath

    Returns:
    X (list): list of messages
    Y (dataframe): disaster categories dataframe
    category_names (list): disaster category names
    """
    # Load dataset from database with read_sql_table
    db = 'sqlite:///' + database_filepath
    engine = create_engine(db)
    df = pd.read_sql_table('DisasterResponse', engine)

    # Define feature and target variables X and Y
    X = df['message']
    Y = df.drop(['message','original','genre','id'], axis=1)
    category_names = Y.columns.values
    return X, Y, category_names

# 3. WRITE A TOKENIZATION FUNCTION TO PROCESS YOUR TEXT DATA

def tokenize(text):
    """Tokenize text messages

    Parameters:
    Text (string): string of text

    Returns:
    clean_token
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# 4. BUILD MACHINE LEARNING PIPELINE

def build_model():
    """Build model: enhanced Randon Forest Classifier ML model,
    with GridSearchCV on vect__ngram_range & tfidf_use_idf parameters

    Returns:
    model: machine learning model
    """
    # enhanced Random Forest Classifier ML model - ran GridSearchCV, found
    # best params for vect_ngram_range=(1,2) & tfidf_use_idf=False

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer(use_idf='False')),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
    'vect__ngram_range': ((1,1),(1,2)),
    'tfidf__use_idf': (True, False)
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model

# 5. PRINT MODEL PERFORMANCE

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model: run evaluation for the output model from build_model() function

    Parameters:
    model: machine learning model output from build_model() function
    X_test (list): test set of messages
    Y_test (dataframe): test set of disaster categories of X_test
    category_names (list): list of disaster category names

    Returns:
    Metrics from classification_report for the built model
    """
    # predict on test data
    y_pred = model.predict(X_test)

    # report the f1 score, precision and recall for each output category of the dataset
    y_test_df = Y_test
    y_pred_df = pd.DataFrame(data=y_pred, columns=category_names)

    for col_name in category_names:
        print(col_name + 'classification_report', classification_report(y_test_df[col_name].values, y_pred_df[col_name].values))
        print('model accuracy mean: ', model.score(X_test, Y_test))

# 6. EXPORT TRAINED MODEL AS A PICKLE FILE

def save_model(model, model_filepath):
    """Save model to pickle file

    Parameters:
    model: machine learning model output from build_model() function
    model_filepath: filepath to store the output pickle file

    Returns: pickle file saved in specified model_filepath
    """
    # save the model to disk
    model_pickle_file = model_filepath
    pickle.dump(model, open(model_pickle_file, 'wb'))

    return model_filepath


def main():
    """
    Run functions
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
