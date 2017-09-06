# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:10:22 2017

@author: sahebsingh
"""

import os
import sys

import scipy as sp

from sklearn.feature_extraction.text import CountVectorizer

#from utils1 import DATA_DIR
#
#TOY_DIR = os.path.join(DATA_DIR, "toy")
#posts = [open(os.path.join(TOY_DIR, f)).read() for f in os.listdir(TOY_DIR)]

path='/Users/sahebsingh/Documents/books/Machine Learning/chap3/toy'

filename=os.listdir(path)
print(filename)
listname=[]
for i in range(1,6):
    listname.append(filename[i])
    
DIR='/Users/sahebsingh/Documents/books/Machine Learning/chap3/toy'
posts = [open(os.path.join(DIR, f)).read() for f in listname]

#import glob
#path='/Users/sahebsingh/Documents/books/Machine Learning/chap3/toy'
#posts=[]
#for filename in glob.glob(os.path.join(path, '*.txt')):
#        with open(filename) as f:
#            for line in f:
#                if line.split()!=0:
#                    posts.append(line.split())
#
#posts=[x for x in posts if len(x)!= 0]
#print(posts)
    
new_post = "imaging databases"

import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')


class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

#vectorizer = CountVectorizer(min_df=1, stop_words='english',\
#preprocessor=english_stemmer)
vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')

from sklearn.feature_extraction.text import TfidfVectorizer


class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(
    min_df=1, stop_words='english', decode_error='ignore')

X_train = vectorizer.fit_transform(posts)

num_samples, num_features = X_train.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))

new_post_vec = vectorizer.transform([new_post])
print(new_post_vec, type(new_post_vec))
print(new_post_vec.toarray())
print(vectorizer.get_feature_names())


def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())


def dist_norm(v1, v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())

    delta = v1_normalized - v2_normalized

    return sp.linalg.norm(delta.toarray())

dist = dist_norm

best_dist = sys.maxsize
best_i = None

for i in range(0, num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist(post_vec, new_post_vec)

    print("=== Post %i with dist=%.2f: %s" % (i, d, post))

    if d < best_dist:
        best_dist = d
        best_i = i

print("Best post is %i with dist=%.2f" % (best_i, best_dist))




















