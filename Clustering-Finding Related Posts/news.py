# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:45:32 2017

@author: sahebsingh
"""

""" In this we will look into a much larger dataset. In this all data of
different topics is already grouped together. """

import sklearn.datasets
import nltk.stem
from stop_words import get_stop_words
import scipy as sp

from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = get_stop_words('english')
english_stemmer = nltk.stem.SnowballStemmer('english')

groups = [
'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']

                                 
#Taking all the Data.And loading it with the help of sklearn.datasets

#all_data = sklearn.datasets.fetch_20newsgroups(subset = "all")
#print(all_data.target_names)


#Taking only Training Data.
train_data = sklearn.datasets.fetch_20newsgroups(subset="train",
                                                 categories=groups)
print(len(train_data.filenames))
print(train_data.target_names)                                                    


#Now we will load test data. 
test_data = sklearn.datasets.fetch_20newsgroups(subset = "test",
                                                categories = groups)
print(len(test_data.filenames))
print(test_data.target_names) 


""" Clustering Posts """
#This data is noisy if we run data as it is it will give UnicodeDecodeError.
#For this we will use StemmedTfidfVectorizer.

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,
                        stop_words='english', decode_error='ignore'
                                    )

x_transform = vectorizer.fit_transform(train_data.data)#Fitting all the data.
num_samples, num_features = x_transform.shape
print("There are %d number of samples and %d number of features" 
        %(num_samples, num_features))


""" We will apply K-Means and keep the number of clusters as 50  """
num_clusters = 50  

from sklearn.cluster import KMeans   

#Initializing KMeans.
km = KMeans(n_clusters = num_clusters, init = 'random', n_init = 1,
            verbose = 1, random_state = 1)

#Using KMeans on x_transform.
km.fit_transform(x_transform)

"""Checkpoint 1"""
""" We have created a vectorizer using KMeans. """

post = """Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks.
"""
new_post_vec = vectorizer.transform([post])
new_post_label = km.predict(new_post_vec)[0]#We will predict it's cluster 
print(new_post_label)                                            #using km.predcit.
# Comparing all 
similar_indices = (km.labels_ == new_post_label).nonzero()[0]


similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec - x_transform[i]).toarray())
    similar.append((dist, train_data.data[i]))

similar = sorted(similar) 
print(len(similar))
show_at_1 = similar[0]
show_at_2 = similar[int(len(similar) / 10)]
show_at_3 = similar[int(len(similar) / 2)]

print("=== #1 ===")
print(show_at_1)
print()

print("=== #2 ===")
print(show_at_2)
print()

print("=== #3 ===")
print(show_at_3)


   
   



















                                                    











