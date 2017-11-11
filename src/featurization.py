from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
from sklearn.feature_extraction.text import HashingVectorizer,CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def term_frequency(doc):
    vect= CountVectorizer(max_df=0.95, min_df=2,)
    X= vect.fit_transform(doc)
    return X.A,vect.get_feature_names()

def tf_idf(doc):
    vect= TfidfVectorizer()
    X= vect.fit_transform(doc)
    return X.A,vect.get_feature_names()

def hashing(doc):
    vect= HashingVectorizer(n_features=1000)
    X= vect.fit_transform(doc)
    return X.A,''

def lda(doc):
    corpus,vocab=term_frequency(doc)
    return LatentDirichletAllocation(n_topics=100,max_iter=1000).fit_transform(corpus), vocab
