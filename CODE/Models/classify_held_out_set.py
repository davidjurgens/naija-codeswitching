import csv
import sys
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

import numpy as np
import scipy

if __name__ == "__main__":

	X = []
	y = []

	with open('/Users/Development/sghosh/Development/NigeriaMediaCorpus/CODE/name_test_data.tsv') as f:
		tsv_reader = csv.reader(f, delimiter="\t")
        next(tsv_reader, None)  # skip the headers

		for annotation in tsv_reader:
			X.append(annotation[0])
			y.append(annotation[2])

	with open('names_classifier.pkl', 'rb') as f:
		model, labels = pickle.load(f)

	y_true = []
	for annotation in y:
		y_true.append(labels.index(annotation.lower()))

	predictions = model.decision_function(X)
	print(roc_auc_score(y, predictions))