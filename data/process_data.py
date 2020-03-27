import sys
import os

import sqlite3
from sqlalchemy import create_engine

import pandas as pd
import numpy as np

def load_data(messages_filepath, categories_filepath):
    """Load data: load disaster messages and disaster categories csv files'

    Parameters:
    messages_filepath (csv file): disaster messages csv file filepath
    categories_filepath (csv file): disaster categories csv filepath

    Returns:
    df: cleaned dataframe

   """
    messages = pd.read_csv(messages_filepath) # load csv file from message_filepath
    categories = pd.read_csv(categories_filepath) # load csv file from categories_filepath
    df = pd.merge(messages, categories, on='id') # merge datasets
    return df


def clean_data(df):
    """Clean data: split categories into separate columns, convert each column
    value to just number 0/1, replace categories columns in df with new columns,
    remove duplicates, and other cleaning

    Parameters:
    df (dataframe): raw dataframe loaded from load_data() function

    Returns:
    df: cleaned dataframe

   """
    ## 1. SPLIT CATEGORIES INTO SEPARATE CATEGORY COLUMNS.

    categories = df.categories.str.split(';', expand=True) # create a dataframe of the 36 individual category columns
    row = categories.iloc[1] # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    # apply a lambda function that takes everything up to the second to last character of each string with slicing
    category_colnames = list(row.apply(lambda cat: cat[:-2]))
    categories.columns = category_colnames # rename the columns of `categories`

    ## 2. CONVERT CATEGORY VALUES TO JUST NUMBERS 0 OR 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda val: val[-1:])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    ## 3. REPLACE CATEGORIES COLUMN IN DF WITH NEW CATEGORY COLUMNS.
    df.drop(['categories'], axis=1, inplace=True) # drop the original categories column from `df`

    df = pd.concat([df, categories], axis=1) # concatenate the original dataframe with the new `categories` dataframe

    ## 4. REMOVE DUPLICATES
    df.drop_duplicates(inplace=True) # drop duplicates

    # 5. OTHER CLEANING
    df.related.replace(2, 1, inplace=True) # category column 'related', replace '2' with '1'

    return df


def save_data(df, database_filepath):
    """Save data: save data to SQLite database file

    Parameters:
    df (dataframe): cleaned dataframe loaded from clean_data() function
    database_filepath (SQLite database .db): output SQLite database filepath

    Output: SQLite database file saved to database_filepath

   """
    # save the clean dataset into an sqlite database
    db = 'sqlite:///' + database_filepath
    engine = create_engine(db)
    df.to_sql('DisasterResponse', engine, index=False)

def main():
    """Run functions

    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
