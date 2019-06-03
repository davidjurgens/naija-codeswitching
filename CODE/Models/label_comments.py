from sklearn.base import BaseEstimator, TransformerMixin

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

import numpy as np
import scipy

from nltk.tokenize import sent_tokenize

import pandas as pd
import pickle

from numpy import random

from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.sparse import csr_matrix

#import stanfordnlp
from collections import Counter
from collections import defaultdict

from sklearn.svm import SVC

from dateutil.parser import parse

import json

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TextNormalizationTransform(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        #print(X)
        output = [re.sub('[^A-Za-z]+', ' ', sentence.lower()) for sentence in X]
        #print('tnt %d' % len(output))
        return output
    
    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        # print(X)        
        return self    


'''
"3484437643": {
            "author": {
                "value": "Odeyale Hakim"
            },
            "depth": 2,
            "message": {
                "value": "<p>You act more like a thug with your comment . When people like you can't make meaningful contribution without throwing barbs , that shows the the kind you are .  Mr educationist , most universities in western world do not need government grants anymore, they source for funding from private sectors. Nigeria universities should learn from that instead of looking for ways to embezzled money. We know  how corrupt ASUU are or you need me to remind you ? Bunch of lazy people , they expect govenrment to provide everything for them. Mind you, I  ain't like you who hasn't gone out of his African locality . Tell your ilks to learn how university needs to run and stop waiting for government handouts . Lazy ass, they dont shown up for lectures , they instruct students to buy their handouts. I bet you know better than former minister of education ?  Stop looking for govenrment handouts , learn how to generate funds .</p>"
            },
            "profile": "https://disqus.com/by/odeyalehakim/",
            "parent": 3483502803,
            "createdAt": "2017-08-24T11:46:50",
            "points": 0
        },
'''

import sys

def main():

    clf_fname = sys.argv[1]
    data_fnames = sys.argv[2:]
    # output_fname = sys.argv[2]
    output_fname = 'comment-classifications.v2.tsv'
    
    with open(clf_fname, 'rb') as f:
        clf = pickle.load(f)

    post_and_comments = {}
    for data_fname in data_fnames:
        with open(data_fname) as f:
            origin = data_fname.split('.')[0].split('/')[-1]
            pc = json.load(f)
            for v in pc.values():
                for v2 in v.values():
                    # print(v)
                    v2['origin'] = origin
            post_and_comments.update(pc)

    #subset = {k: post_and_comments[k] for k in list(post_and_comments)[:5000]}
    #post_and_comments = subset
            
    tribe_names = set()
    with open('tribe-list.txt') as f:
        for line in f:
            tribe_names.add(line[:-1].lower())
            
    print("loaded comments for %d posts" % len(post_and_comments))

    df = defaultdict(list)
    '''{
        'comment_id': [], 'author': [], 'depth': [], 'message': [],
        'has_naija': [], 'date': [], 'parent_id': [],
        'points': [], 'day_of_week': [], 'day_type': [],
        'author_url': [], 'sentiment': [], 'origin': [],
        }'''

    s2c = defaultdict(list)
    analyser = SentimentIntensityAnalyzer()

    id_to_score = {}
    id_to_text = {}
    
    snum =  0
    ng_num = 0
    for post, comments in post_and_comments.items():
        for comment_id, content in comments.items():
            # print(content)
            #try:
            text = content['message']['value']
            #except:
            #    print(content)
            text = re.sub('<[^>]+>', ' ', text)
            sentences = sent_tokenize(text)
            has_naija = False
            for sentence in sentences:
                #print(len(sentence))
                sentence = sentence.strip()
                # Skip short sentences, which we're bad at
                if len(sentence) < 5:
                    continue
                ng_prob = clf.predict_proba([sentence,])[0][0]
                if True:
                    s = int(ng_prob * 10)
                    s2c[s].append('%f\t%s\n' % (ng_prob, sentence))
                if ng_prob >= 0.5:
                    #print(sentence, ng_prob)
                    has_naija = True
                    ng_num += 1
                    break

            snum += 1
            if snum % 10000 == 0:
                print('saw %d comments, %d Naija' % (snum, ng_num))

            # TODO: sentiment stuff
            polarity_score = analyser.polarity_scores(text)['compound']
            id_to_score[comment_id] = polarity_score
            
            # TODO: date stuff
            date = parse(content['createdAt'])
            month = date.month
            day_of_week = date.weekday() 
            if day_of_week < 6:
                day_type = "weekday"
            else:
                day_type = "weekend"
            year = date.year

            id_to_text[comment_id] = text

            parent_id = str(content['parent'])
            
            df['comment_id'].append(comment_id)
            df['author'].append(content['author']['value'])
            df['depth'].append(content['depth'])
            df['message'].append(text)
            df['has_naija'].append(1 if has_naija else 0)

            # Sometimes at depth 0, the parent is listed at this ID
            if parent_id == comment_id:
                df['parent_id'].append(-1)
            else:
                df['parent_id'].append(parent_id)
                
            df['points'].append(content['points'])
            df['date'].append(content['createdAt'])
            df['day_of_week'].append(day_of_week)
            df['day_type'].append(day_type)
            df['author_url'].append(content['profile'])
            df['sentiment'].append(polarity_score)
            df['origin'].append(content['origin'])
            df['year'].append(year)
            df['month'].append(month)

    lines = []
    for s, cs in s2c.items():
        # print(s, len(cs), cs[:2])        
        for c in random.choice(cs, min(500, len(cs)), replace=False):
            lines.append(c)

    with open('naija-comments.sample.tsv', 'wt') as outf:
        outf.write('label\tcomment\n')
        for line in lines:
            outf.write(line)
   
    # Make a second pass to fill in stuff about parents
    for parent_id in df['parent_id']:
        if parent_id not in id_to_score:
            df['parent_sentiment'].append(0)
        else:
            df['parent_sentiment'].append(id_to_score[parent_id])

        has_tribe = 0
        if parent_id in id_to_text:
            parent_text = id_to_text[parent_id]
            # See if it mentions some tribe
            words = set(re.sub('[^a-z]+', ' ', parent_text.lower()).split())
            if not tribe_names.isdisjoint(words):
                has_tribe = 1
        df['parent_mentions_tribe'].append(has_tribe)

    for k, v in df.items():
        print('%s\t%d' % (k, len(v)))
    df = pd.DataFrame(df)
    df.to_csv(output_fname, sep='\t', index=False)
            
if __name__ == '__main__':
    main()
