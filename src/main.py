from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import pandas as pd
from Preprocess import *
import numpy as np
from featurization import *
from random import seed

def processing(x):
    return process(x, string_lower, email_urls, punctuate_preproc,
            numeric_isolation, stopwords, stemming, word_len_less)

def spam_preprocess():
    path='../data/spam.csv'
    df=pd.read_csv(path)
    df.drop(df.columns[2:],axis=1,inplace=True)
    df['v2']=df['v2'].apply(lambda x: processing(x))
    df.to_csv('../data/processed_spam.csv', index=False)
    print(df)

def readcsv():
    path='../data/processed_spam.csv'
    df=pd.read_csv(path)
    return np.array(df[df.columns[0]].values.tolist()), np.array(df[df.columns[1]].values.tolist())

if __name__ == '__main__':
    seed(1)
    np.random.seed(1)
    ## preprocessing
    # spam_preprocess()

    labels, corpus=readcsv()

    ## Featurizatoin method
    features=[term_frequency,tf_idf,hashing,lda]

    for i in features[:1]:
        corpus,vocab=i(corpus)


