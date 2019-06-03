import pickle
import os
import glob
import sys

from sklearn.model_selection import train_test_split

import re
import numpy as np
import pandas as pd

import csv
import unicodedata
import string
import json

all_letters = string.ascii_letters + " .,;'"
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

'''
run with 1 argument:
(1) directory containing comment ddata
'''

if __name__ == "__main__":
        comment_directory = sys.argv[1]
        websites = glob.glob(comment_directory+'/*.json')

        with open('names_classifier.pkl', 'rb') as f:
                model, labels = pickle.load(f)

        print(labels)
        random_names = []
        names_heldout = []

        names = []

        for website in websites:
                current_names = []
                with open(website, 'r') as f:
                        comments = json.load(f)

                        for article,comment_data in comments.items():
                                if comment_data:
                                        for post, post_data in comment_data.items():
                                                name = post_data["author"]["value"]
                                                name = unicodeToAscii(name).lower()
                                                current_names.append(name)

        current_names = list(set(list(current_names)))

        other_names, held_out_sample = train_test_split(names, test_size=0.1, random_state=0)
        
        predictions = model.predict_proba(current_names)
        class_predictions = model.predict(current_names)
        
        random_sample = np.random.choice(len(current_names), 250)
        random_names.extend(zip(np.array(current_names)[random_sample], np.array(class_predictions)[random_sample]))
        
        names_heldout.extend(held_out_sample)
        
        names.extend(zip(list(current_names), list(class_predictions)))
        print(names[:10])


        '''
        with open('random_usernames.csv', 'w') as f:    
                random_names_with_class = []
                for name in random_names:
                        random_names_with_class.append((name[0], labels[name[1]]))
                writer = csv.writer(f, delimiter=',')
                for prediction in random_names_with_class:
                        writer.writerow(prediction)

        with open('held_out_set.txt', 'w') as f:
                random_heldout = np.random.choice(names_heldout, 200)
                for name in random_heldout:
                        f.write(name+'\n')
        '''

        with open('username_classifications.tsv', 'w') as f:
            usernames_with_class = []
            for name in names:
                try:
                    usernames_with_class.append((name[0], labels[name[1]]))
                except:
                    print(name[0], name[1])
                    break
            writer = csv.writer(f, delimiter='\t')
            for prediction in usernames_with_class:
                writer.writerow(prediction)

