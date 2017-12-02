from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from hmmlearn.hmm import GaussianHMM

def run_dt(train_data,train_labels,test_data,test_labels):
    model = DecisionTreeClassifier().fit(train_data, train_labels)
    prediction=model.predict(test_data)
    return f1_score(test_labels, prediction,pos_label='spam')

def run_rf(train_data,train_labels,test_data,test_labels):
    model = RandomForestClassifier().fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return f1_score(test_labels, prediction,pos_label='spam')

def run_svmlinear(train_data,train_labels,test_data,test_labels):
    model = SVC(kernel='linear').fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return f1_score(test_labels, prediction,pos_label='spam')

def run_svmrbf(train_data,train_labels,test_data,test_labels):
    model = SVC(kernel='rbf').fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return f1_score(test_labels, prediction,pos_label='spam')

def naive_bayes(train_data,train_labels,test_data,test_labels):
    model = GaussianNB().fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return f1_score(test_labels, prediction,pos_label='spam')

def log_reg(train_data,train_labels,test_data,test_labels):
    model = LogisticRegression().fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return f1_score(test_labels, prediction,pos_label='spam')

def knn(train_data,train_labels,test_data,test_labels):
    model = KNeighborsClassifier(n_neighbors=8).fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return f1_score(test_labels, prediction,pos_label='spam')

def hmm(train_data,train_labels,test_data,test_labels):
    model = GaussianHMM(n_components=2, covariance_type="full").fit(train_data)
    prediction = model.predict(test_data)
    return f1_score(test_labels, prediction,pos_label='spam')

