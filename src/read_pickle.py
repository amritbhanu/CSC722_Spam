from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import pickle
import numpy as np
import pandas as pd

with open("../dump/spam_nosmote.pickle", 'rb') as handle:
    nosmote = pickle.load(handle)
with open("../dump/spam_yessmote.pickle", 'rb') as handle:
    smote = pickle.load(handle)

headers=["features"]+nosmote["term_frequency"].keys()
temp=[]
for i in nosmote.keys():
    l=[]
    l.append(i)
    for a,b in nosmote[i].iteritems():
        l.append(round(np.median(b),2))
    temp.append(l)

df=pd.DataFrame(temp,columns=headers)
df.to_csv("../results/nosmote.csv",index=False)

headers = ["features"] + smote["term_frequency"].keys()
temp = []
for i in smote.keys():
    l = []
    l.append(i)
    for a, b in smote[i].iteritems():
        l.append(round(np.median(b),2))
    temp.append(l)

df = pd.DataFrame(temp, columns=headers)
df.to_csv("../results/smote.csv",index=False)