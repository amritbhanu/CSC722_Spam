from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import pickle

with open("../dump/spam_nosmote.pickle", 'rb') as handle:
    l = pickle.load(handle)

print(l)