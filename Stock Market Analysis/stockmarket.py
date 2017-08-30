# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 21:29:51 2017

@author: sahebsingh
"""

""" We are going to do Stock Market prediction """

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# Import File using Pandas.

file = pd.read_csv('/Users/sahebsingh/Documents/books/Machine Learning/chap4/stocknews/Combined_News_DJIA.csv')

#print(file.head())

# We will split our data into two parts. 

train1, test1 = train_test_split(file, test_size = 0.2)
#print(train1.size)
#print(test1.size)
train = file[file["Date"] > '2014-12-31']
test = file[file["Date"] > '2015-01-01']

""" Checkpoint 1 """

## Text Preprocessing.

example  = train.iloc[3,12]
#print(example)

example2 = example.lower()
#print(example2)

""" Now we will perform text modelling.  """
trainheadlines = []

for row in range(0, len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))

basic_vectorizer = CountVectorizer()
x_transform = basic_vectorizer.fit_transform(trainheadlines)

#print(x_transform.shape)

basicmodel = LogisticRegression()
basicmodel = basicmodel.fit(x_transform, train["Label"])

""" Checkpoint 2 """

""" We will now predict the data. """

testheadlines = []

for row in range(0, len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row, 2:27]))

#print(testheadlines[0])

y_transform = basic_vectorizer.transform(testheadlines)

result = basicmodel.predict(y_transform)

cross = pd.crosstab(test["Label"], result, colnames = ["Predictions"],\
             rownames = ["Actual"])


""" Now we will use n-gram model """

advanced_vectorizer = CountVectorizer(ngram_range = (2, 2))
advancedtrain = advanced_vectorizer.fit_transform(trainheadlines)

#print(advancedtrain.shape)

advancedmodel = LogisticRegression()
advancedtrain = advancedmodel.fit(advancedtrain, train["Label"])

advancedtest = advanced_vectorizer.transform(testheadlines)

newresult = advancedmodel.predict(advancedtest)

crossadvanced = pd.crosstab(test["Label"], newresult, colnames = ["Predictions"],\
                        rownames = ["Actual"])

#print(crossadvanced)


















