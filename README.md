# Disaster Response Pipeline Project

### Project summary:
This project is a part of Udacity's Data Science Nanodegree, the Disaster Response Project - aiming to help students get familiar with and practice their skills in Data Engineering (ETL and Machine Learning pipeline) to build a model for an API that classifies disaster messages.

The project also includes a web app for emergency worker to input messages to be classified into different disaster categories, as well as visualisations to showcase the messages and their disaster categories in the database.

### Files in the repository:
In the data folder:
- disaster_categories.csv: disaster categories data to be processed in the process_data.py
- disaster_messages.csv: disaster messages data to be processed in the process_data.py
- DisasterResponse.db: cleaned & transformed data SQLite database, output from the process_data.py
- ETL Pipeline Preparation.ipynb: Jupyter Notebook used to prepare for the ETL pipeline
- process_data.py: Python scripts which is translated from the ETL Pipeline Preparation notebook

In the models folder:
- ML Pipeline Preparation.ipynb: Jupyter Notebook used to prepare for the Machine Learning pipeline
- train_classifier.py: Python scripts which is translated from the ML Pipeline Preparation notebook
- classifer.pkl: pickle file from the Machine Learning model, output from the train_classifier.py

In the app folder:
- run.py: Flask file used to run app
- templates/go.html: disaster classification result page after the emergency worker input messages
- templates/master.html

### Version requirements
- Scikit-learn version = 0.21.2
- Python version = 3.6.3

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
