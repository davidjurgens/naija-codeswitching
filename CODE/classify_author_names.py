import pickle
import os
import glob
import sys

import re
import numpy as np
import pandas as pd

import csv
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

'''
run with 1 argument:
(1) directory containing article data
'''

if __name__ == "__main__":
	article_directory = sys.argv[1]
	websites = glob.glob(article_directory+'/*.tsv')

	stop_words = ['assistant', 'editor', 'asst.', 'editor', 'reporter', 'reports', 'by', 'from', 'abuja', 'lagos', 'benin city', 'our', 'dr', 'correspondent', 'rev.', 'mr.', 'author', ';']

	with open('names_classifier.pkl', 'rb') as f:
		model, labels = pickle.load(f)

	names = []
	uncertain_predictions = []
	random_names = []

	for website in websites:
		current_names = []

		print(website)
		if os.path.splitext(os.path.basename(website))[0] != 'punch_articles':

			with open(website, 'r') as f:
				tsv_reader = csv.reader(f, delimiter="\t")
				for article in tsv_reader:
					name = article[3]
					name = unicodeToAscii(name).lower()
					for word in stop_words:
						name = name.replace(word, ' ')
					name_list = name.split('and')
					current_names.extend([name.strip() for name in name_list])

				current_names = list(set(list(current_names)))
				predictions = model.predict_proba(current_names)
				class_predictions = model.predict(current_names)


				current_names, held_out_sample = train_test_split(names, test_size=0.1, random_state=0)

				names_heldout.extend(held_out_sample)
				
				random_sample = np.random.choice(len(current_names), 250)
				random_names.extend(zip(np.array(current_names)[random_sample], np.array(class_predictions)[random_sample]))

				'''
				predictions.sort(axis=1)			
						
				ratios = []
						
				for prediction in predictions:
					ratios.append(prediction[2]/prediction[1])
						
				ratios = sorted(enumerate(ratios), key=lambda x: x[1])
						
				for i in range(0, 100):
					uncertain_predictions.append((names[ratios[i][0]], labels[class_predictions[ratios[i][0]]]))

				'''


						
				'''
				print(os.path.splitext(os.path.basename(website))[0]+'\n')
				print("{}: {} / {} \n".format(labels[0], len([i for i in predictions if i == 0]), len(predictions)))
				print("{}: {} / {} \n".format(labels[1], len([i for i in predictions if i == 1]), len(predictions)))
				print("{}: {} / {} \n".format(labels[2], len([i for i in predictions if i == 2]), len(predictions)))
				'''
			names.extend(zip(np.array(current_names), np.array(class_predictions)))

	'''
	with open('random_names.csv', 'w') as f:	
		random_names_with_class = []
		for name in random_names:
			random_names_with_class.append((name[0], labels[name[1]]))
		writer = csv.writer(f, delimiter=',')
		for prediction in random_names_with_class:
			writer.writerow(prediction)
	'''

	with open('held_out_set.txt', 'w') as f:
		random_heldout = np.random.choice(names_heldout, 200)
		for name in random_heldout:
			f.write(name+'\n')
			
	with open('author_classifications.csv', 'w') as f:
		names_with_class = []
		for name in names:
			names_with_class.append((name[0], labels[name[1]]))
		writer = csv.writer(f, delimiter=',')
		for prediction in names_with_class:
			writer.writerow(prediction)
