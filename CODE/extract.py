import csv
import sys
import pickle

with open('heldoutdata.pkl', 'rb') as f:
	X_heldout, y_heldout = pickle.load(f)

with open('heldoutdata.txt', 'w') as f:
	for example in X_heldout:
		f.write(example+'\n')

with open('heldoutclasses.txt', 'w') as f:
	for classification in y_heldout:
		f.write(classification)
		f.write('\n')