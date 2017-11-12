from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import pandas as pd
from Preprocess import *
import numpy as np
from featurization import *
from random import seed
from sklearn.model_selection import StratifiedKFold
from ML import *
import pickle

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
    learners=[run_dt,run_rf,run_svmlinear,run_svmrbf,log_reg,knn,naive_bayes]
    final={}
    for i in features[:1]:
        print(i.__name__)
        temp={}
        corpus,vocab=i(corpus)
        skf = StratifiedKFold(n_splits=10)
        for k in learners[:1]:
            l=[]
            print(k.__name__)
            for train_index, test_index in skf.split(corpus, labels):
                train_data, train_labels = corpus[train_index], labels[train_index]
                test_data, test_labels = corpus[test_index], labels[test_index]
                value=k(train_data, train_labels, test_data, test_labels)
                l.append(value)
            temp[k.__name__]=l
        final[i.__name__]=temp
    with open('../dump/spam_nosmote.pickle', 'wb') as handle:
        pickle.dump(final, handle)