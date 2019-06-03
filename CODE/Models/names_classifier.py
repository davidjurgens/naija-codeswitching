from __future__ import unicode_literals
import os
import glob
import sys

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier

from sklearn.model_selection import cross_val_score

import numpy as np
import scipy

from nltk.tokenize import sent_tokenize

import pandas as pd
import pickle

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

if __name__ == "__main__":

	X = []
	y = []

	labels = []
	i = 0

	for file in glob.glob('Name_recognition/*.txt'):
		tribe = os.path.splitext(os.path.basename(file))[0]

		with open(file, 'r') as f:
			names = [line.rstrip('\n') for line in f]

		labels.append(tribe)
		if tribe == 'yoruba':
			names = np.random.choice(names, 500)

		for name in names:
			X.append(unicodeToAscii(name.lower()))
			y.append(i)
		i += 1


	'''
	# create classifier for each word n-gram for n in range [1, 4)
	for i in range(1, 5):
	    vectorizer = CountVectorizer(ngram_range=(2,5), analyzer='char')

	    X_train = vectorizer.fit_transform(X)
	    clf = LogisticRegression(solver='lbfgs', multi_class='ovr')
	    scores = cross_val_score(clf, X_train, y, cv=5, verbose=100)

	    print("{} char grams".format(i))
	    print(scores.mean())
	'''

	vectorizer = CountVectorizer(ngram_range=(2,5), analyzer='char')
	clf = LogisticRegression(solver='lbfgs', multi_class='ovr')


	pipe = Pipeline([('vectorizer', vectorizer), ('lrg', clf)])
	pipe.fit(X, y)

	dummy = DummyClassifier(strategy='most_frequent', random_state=0)
	dummy.fit(X, y)

	with open('names_classifier.pkl', 'wb') as f:
 		pickle.dump((pipe, labels, dummy), f)