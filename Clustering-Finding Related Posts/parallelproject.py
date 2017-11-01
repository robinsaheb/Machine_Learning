# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 00:15:44 2017

@author: sahebsingh
"""
import time
import threading
import sklearn.datasets
import nltk.stem
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans 
import scipy as sp

stop_words = get_stop_words('english')
english_stemmer = nltk.stem.SnowballStemmer('english')

start = time.time()

def task1():
    groups = [
'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
    
    global train_data 
    
    train_data = sklearn.datasets.fetch_20newsgroups(subset="train",
                                                 categories=groups)
    print(len(train_data.filenames))
    
    train_data = sklearn.datasets.fetch_20newsgroups(subset="train",
                                                 categories=groups) 
                                            
    print("Task 1 is completed") 
                                           
    return train_data    
    
#    print("Task 1 is completed")

    
def task2():
    groups = [
'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
    
    test_data = sklearn.datasets.fetch_20newsgroups(subset = "test",
                                                categories = groups)
                                            
    print(len(test_data.filenames))
    
    test_data = sklearn.datasets.fetch_20newsgroups(subset = "test",
                                                categories = groups)
                                                

    print("Task 2 is completed")
    
    
def task3():
    
    class StemmedTfidfVectorizer(TfidfVectorizer):
        def build_analyzer(self):
            analyzer = super(TfidfVectorizer, self).build_analyzer()
            return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

    vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,
                        stop_words='english', decode_error='ignore'
                                    )
    x_transform = vectorizer.fit_transform(train_data.data)
    num_samples, num_features = x_transform.shape
    print("There are %d number of samples and %d number of features" 
        %(num_samples, num_features))
    

    post = """Disk drive problems. Hi, I have a problem with my hard disk. 
    After 1 year it is working only sporadically now. 
    I tried to format it, but now it doesn't boot any more.
    Any ideas? Thanks.
    """
    num_clusters = 50 

    km = KMeans(n_clusters = num_clusters, init = 'random', n_init = 1,
            verbose = 1, random_state = 1)
    
    km.fit_transform(x_transform)
    new_post_vec = vectorizer.transform([post])
    new_post_label = km.predict(new_post_vec)[0]
    print(new_post_label)
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
          
    print("Task 3 is completed")   
    
    print(show_at_3)
  
    
def dep1():
    t1 = threading.Thread(target = task1)
    t2 = threading.Thread(target = task2)
    t3 = threading.Thread(target = task3)
    t1.start()
    t2.start()
    t1.join()
    t3.start()
    t2.join()
    t3.join()
    
dep1()

end = time.time()

print(" ")
print(" ")
print("This is the total time taken in seconds to execute the parallel Program", end - start)




















    
















