# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 01:15:48 2023

@author: sbhadra
"""
import pandas as pd
import re
import numpy as np
import streamlit as st

import csv
import pickle
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

nltk.download('stopwords')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

#import s3fs
#import os

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

# items to be removed
unwanted_words = {'no', 'nor', 'not','don', "don't",'ain', 'aren', "aren't",
                  'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
                  'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
                  'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                  "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                  "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                  "won't", 'wouldn', "wouldn't"}
 
NEW_STOPWORDS = [ele for ele in STOPWORDS if ele not in unwanted_words]

#@st.cache_resource
#def read_file():
#    filename = "s3://ordsmall/finalized_model.sav"
#    loaded_model = pickle.load(open(filename, 'rb'))
#    return loaded_model
    #with fs.open(filename) as f:
    #    return pickle.load(f)

#loaded_model = read_file()

#@st.cache_resource
#def load_model():
#    filename = 'finalized_model.sav'

    # Load the saved model
 #   loaded_model = pickle.load(open(filename, 'rb'))
 #   return loaded_model

# Load the model
#loaded_model = load_model()

#C = loaded_model['clf'].coef_

@st.cache_data
def load_data():
    # filename = 'finalized_model.csv'
    filename = 'olr_model.csv'
    
    # Load the saved data
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        trained_dictionary = []
        trained_idf = []
        fitted_coef = []
    
        for row in reader:
            trained_dictionary.append(row[0])
            trained_idf.append(float(row[1]))
            fitted_coef.append(float(row[2]))
        
    # Convert num_array to a numpy array
    trained_idf = np.array(trained_idf)
    fitted_coef = np.array(fitted_coef)
    
    return trained_dictionary, trained_idf, fitted_coef

# Load the data
trained_dictionary, trained_idf, fitted_coef = load_data()

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in NEW_STOPWORDS) # delete stopwors from text
    return text

# Define the predict_severity function that takes user input and returns a prediction
#def predict_severity(text):
#    D = loaded_model['tfidf'].transform(loaded_model['vect'].transform(pd.Series(clean_text(text))))
#    Ystar = D.dot(C)
    # Use the pipeline to predict the number
    #predicted = loaded_model.predict([text])
#    return Ystar[0]

def predict_severity(text):
    cvt = CountVectorizer(ngram_range=(1, 4), vocabulary = trained_dictionary)
    Ystar = normalize(cvt.transform(pd.Series(clean_text(text))).multiply(trained_idf), norm = 'l2', axis = 1).dot(fitted_coef)
    
    return Ystar[0]

# Create the Streamlit app
def main():
    # Set the title and description of the app
    st.title('Text Severity Predictor')
    st.write('Enter some text below to predict the severity')

    # Create a text box for user input
    user_input = st.text_input('Enter text here')

    # Create a button to trigger the prediction
    #if st.button('Find Severity'):
        # If the user has entered some text, run the prediction function
    if user_input:
        prediction = predict_severity(user_input)
        
        # 1.83072462 11.33662054 for olr_model
        if prediction < 1.83072462:
            category = 'Malfunction'
        elif prediction < 11.33662054:
            category = 'Injury'
        else:
            category = 'Death'
        
        # Display the predicted severity
        st.write('Predicted Severity:', prediction)
        st.write('Category:', category)
        
# Run the app
if __name__ == "__main__":
    main()
