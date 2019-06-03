import os
import glob
import sys

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import numpy as np
import scipy

from nltk.tokenize import sent_tokenize

import pandas as pd
import pickle
import json

'''
run with 1 argument:
(1) directory containing comment json
'''

if __name__ == "__main__":

	comments_directory = sys.argv[1]

	test_data = []

	for file in glob.glob(comments_directory+"*.json"):
		with open(file, 'r') as f:
			comment_data = json.load(f)
			for uri, comments in comment_data.items():
				if comments:
					for post_id, values in comments.items():
						comment_string = values["message"]["value"]
						comment_sentences = sent_tokenize(comment_string.replace('.', '. '))
						comment_sentences = [re.sub('[^A-Za-z]+', ' ', sentence.lower()) for sentence in comment_sentences]
						for sentence in comment_sentences:
							test_data.append(sentence)

	with open('logistic_model.pkl', 'rb') as f:
		model = pickle.load(f)

	y_predicted = model.decision_function(test_data)

	sorted_predictions = sorted(enumerate(y_predicted), key = lambda x: x[1])

	lowest_predicted = sorted_predictions[0:100]
	highest_predicted = sorted_predictions[-100:]

	with open('sorted_results.txt', 'w+') as f:
		for sentence in lowest_predicted:
			f.write((test_data[sentence[0]]))

		for sentence in highest_predicted:
			f.write((test_data[sentence[0]]))
