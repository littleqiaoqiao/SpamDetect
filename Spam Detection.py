import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import csv
import sklearn
# Read the csv file
df = pd.read_csv('spam.csv',encoding = "ISO-8859-1")

#Print the first 5 rows of data
df.head(5)

#Print the shape (Get the number of rows and columns)
df.shape

#Get the column names
df.columns

#Show the number of missing (NAN, NaN, na) data for each column
df.isnull().sum()

def process_text(text):
    #1 remove punctuation
    #2 remove stopwords
    #3 return a list of clean text words

    #1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    #3
    return clean_words

#Show the tokenization (a list of tokens also called lemmas)
df['v2'].head().apply(process_text)

#Convert a collection of text to a matrix of tokens
from sklearn.feature_extraction.text import CountVectorizer
message_bow = CountVectorizer(analyzer=process_text).fit_transform(df['v2'])

#Split the data into 80% training and 20% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(message_bow, df['v1'], test_size=0.20, random_state=0)

#Get the shape of message_bow
print(message_bow.shape)

#Create and train the Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(X_train, y_train)

#Print the predictions
print(classifier.predict(X_train))

#Print the actual values
print(y_train.values)

#Evaluate the model on the training data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(X_train)
print(classification_report(y_train, pred))
print()
print('Confusion Matrix: \n', confusion_matrix(y_train, pred))
print()
print('Accuracy: \n', accuracy_score(y_train, pred))

#Print the predictions
print(classifier.predict(X_test))

#Print the actual values
print(y_test.values)

#Evaluate the model on the testing data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(X_test)
print(classification_report(y_test, pred))
print()
print('Confusion Matrix: \n', confusion_matrix(y_test, pred))
print()
print('Accuracy: \n', accuracy_score(y_test, pred))
