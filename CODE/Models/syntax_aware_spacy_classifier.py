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

from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.sparse import csr_matrix

#import stanfordnlp
from collections import Counter
import spacy

from sklearn.svm import SVC

class SampleExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, vars):
        self.vars = vars  # e.g. pass in a column name to extract

    def transform(self, X, y=None):
        return do_something_to(X, self.vars)  # where the actual feature extraction happens

    def fit(self, X, y=None):
        return self  # generally does nothing

class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def average_word_length(self, name):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in name.split()])

    def transform(self, X, y=None):

        # Ugh.  All this stupid fancy numpy stuff just to get the damned matrix
        # dimensions to line up :(
        output = np.asmatrix(np.array([self.average_word_length(x) for x in X])).T
        # print('awl', output.shape)
        return output

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class SyntaxFeatureExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        self.nlp = spacy.load('en')
        self.pos_to_index = {}
        self.dep_to_index = {}

        # This is the union of the two so we can keep a consistent feature index
        # ordering.  We use separated data structures though for telling what is
        # what (hopefully) later during debugging
        self.feat_to_index = {}        

    def fit_transform(self, X, y=None):
        print('SFE fit-transform called')
        #X_ = []
        row = []
        col = []
        val = []
        for r, text in enumerate(X):
            if (r+1) % 100 == 0:
                print('parsing sentence %d/%d' % (r+1, len(X)))
            doc = self.nlp(text)
            pos_counts = Counter()
            dep_counts = Counter()
            
            for sent in doc.sents:
                tokens, pos_tags, dep_parse_triples \
                    = self.to_parsed_representations(sent)
                
                for pos in pos_tags:
                    pos_counts[pos] += 1
                for triple in dep_parse_triples:
                    dep = triple[1]
                    dep_counts[dep] += 1
                    
            for pos, count in pos_counts.items():
                
                if pos not in self.pos_to_index:
                    c = len(self.feat_to_index)
                    self.pos_to_index[pos] = c
                    self.feat_to_index[pos] = c
                else:
                    c = self.pos_to_index[pos]

                row.append(r)
                col.append(c)
                val.append(count)

            for dep, count in dep_counts.items():
                
                if dep not in self.dep_to_index:
                    c = len(self.feat_to_index)
                    self.dep_to_index[dep] = c
                    self.feat_to_index[dep] = c
                else:
                    c = self.dep_to_index[dep]

                row.append(r)
                col.append(c)
                val.append(count)


        #print('%d + %d ?= %d' % (len(self.dep_to_index),
        #                         len(self.pos_to_index), len(self.feat_to_index)))
                
                
        output = np.zeros((r+1, len(self.feat_to_index)))
        for i, r in enumerate(row):
            output[r, col[i]] = val[i]
        # print('fit_transform', output.shape)
                    
        #print(self.dep_to_index)
        #print(self.pos_to_index)

        return output

    def to_parsed_representations(self, spacy_sent):
        def get_triples(n, triples):
            n_tag = n.tag_
            if len(n_tag) == 0:
                n_tag = ' '
            elif n_tag == 'BES':
                n_tag = 'VBE'
            elif n_tag == 'HVS':
                n_tag == 'VHV'
            elif n_tag == 'NFP':
                n_tag == 'Punct'
                
            for c in n.children:
                c_tag = c.tag_
                if len(c_tag) == 0:
                    c_tag = ' '
                elif c_tag == 'BES':
                    c_tag = 'VBE'
                elif c_tag == 'HVS':
                    c_tag == 'VHV'
                elif c_tag == 'NFP':
                    c_tag == 'Punct'
                
                t = ((n.orth_, n_tag), c.dep_, (c.orth_, c_tag))
                if len(c.dep_) > 0:
                    triples.append(t)
                get_triples(c, triples)
                
        triples = []
        get_triples(spacy_sent.root, triples)
    
        tokens = []
        pos_tags = []
        for t in spacy_sent:
            if t.orth_ == ' ':
                continue
            tokens.append(t.orth_)
            t_tag = t.tag_
            if len(t_tag) == 0:
                t_tag = ' '
            elif t_tag == 'BES':
                t_tag = 'VBE'
            elif t_tag == 'HVS':
                t_tag == 'VHV'
            elif t_tag == 'NFP':
                t_tag == 'Punct'
            
            pos_tags.append(t_tag)
        return tokens, pos_tags, triples
        
        
    def transform(self, X, y=None):
        print('SFE transform called')
        #X_ = []
        row = []
        col = []
        val = []
        for r, text in enumerate(X):
            doc = self.nlp(text)
            pos_counts = Counter()
            dep_counts = Counter()
            
            for sent in doc.sents:
                tokens, pos_tags, dep_parse_triples \
                    = self.to_parsed_representations(sent)
                
                for pos in pos_tags:
                    pos_counts[pos] += 1
                for triple in dep_parse_triples:
                    dep = triple[1]
                    dep_counts[dep] += 1

            for pos, count in pos_counts.items():
                
                if pos in self.pos_to_index:
                    c = self.pos_to_index[pos]
                else:
                    continue
                    
                row.append(r)
                col.append(c)
                val.append(count)


            for dep, count in dep_counts.items():
                
                if dep in self.dep_to_index:
                    c = self.dep_to_index[dep]
                else:
                    continue

                row.append(r)
                col.append(c)
                val.append(count)
                
        # NOTE: we need to manually set the shape here to account for features
        # that weren't present (assuming fit had been called prior)

        print(r, c, len(self.feat_to_index), row, col)
        
        output = np.zeros((r+1, len(self.feat_to_index)))
        for i, r in enumerate(row):
            output[r, col[i]] = val[i]

        #print('transform', output.shape)
                    
        return output

    
    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        print('fit called alone???')
        return self

    
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
run with 2 arguments:
(1) pidgin data
(2) english data
'''

def main():

    
    pidgin_data = sys.argv[1]
    english_data = sys.argv[2]

    pidgin_str = ''
    english_str = ''
    
    labels = ['pi', 'en']

    max_lines = 100000
    pidgin_lines = []
    with open(pidgin_data) as f:
        for line_no, line in enumerate(f):
            pidgin_lines.append(line)
            if line_no >= max_lines:
                break
            
    english_lines = []
    with open(english_data) as f:
        for line_no, line in enumerate(f):
            english_lines.append(line)
            if line_no >= max_lines:
                break


    
    # tokenize each of the texts into sentences
    pidgin_sentences = sent_tokenize(' '.join(pidgin_lines).replace('.', '. '))
    english_sentences = sent_tokenize(' '.join(english_lines).replace('.', '. '))

    # remove all non-alphabetic characters and lowercase eech sentence
    #pidgin_sentences = [re.sub('[^A-Za-z]+', ' ', sentence.lower()) for sentence in pidgin_sentences]
    #english_sentences = [re.sub('[^A-Za-z]+', ' ', sentence.lower()) for sentence in english_sentences]


    training_data = []
    training_targets = []

    n = min(100000, min(len(pidgin_sentences), len(english_sentences)))
    print('Saw %d training instances' % n)
    
    for sentence in pidgin_sentences[:n]:
        training_data.append(sentence)
        training_targets.append(0)

    for sentence in english_sentences[:n]:
        training_data.append(sentence)
        training_targets.append(1)

    pipe = Pipeline([
        ('feature-extraction', FeatureUnion([
            ('cgrams', Pipeline([
                ('normalize', TextNormalizationTransform()), 
                ('vect', CountVectorizer(ngram_range=(2,5), analyzer='char')),
                ('tfidf', TfidfTransformer(use_idf=False)),
            ])),
            ('ngrams', Pipeline([
                ('normalize', TextNormalizationTransform()), 
                ('vect', CountVectorizer(ngram_range=(1,3))),
                ('tfidf', TfidfTransformer(use_idf=False)),
            ])),            
            ('word-length', AverageWordLengthExtractor()),
            ('syntax-feats', SyntaxFeatureExtractor())
        ])),
        ('classifier', LogisticRegression(solver='lbfgs', max_iter=10000))])

    feat_pipe = Pipeline([
        ('feature-extraction', FeatureUnion([
            ('cgrams', Pipeline([
                ('normalize', TextNormalizationTransform()), 
                ('vect', CountVectorizer(ngram_range=(4,4), analyzer='char')),
                ('tfidf', TfidfTransformer(use_idf=False)),
            ])),
            ('word-length', AverageWordLengthExtractor()),
            ('syntax-feats', SyntaxFeatureExtractor())
        ])),
    ])

    syntax_feat_pipe = Pipeline([
        ('feature-extraction', FeatureUnion([
            ('syntax-feats', SyntaxFeatureExtractor())
        ])),
    ])
    
    surface_feat_pipe = Pipeline([
        ('feature-extraction', FeatureUnion([
            ('cgrams', Pipeline([
                ('normalize', TextNormalizationTransform()), 
                ('vect', CountVectorizer(ngram_range=(2,5), analyzer='char')),
                ('tfidf', TfidfTransformer(use_idf=False)),
            ])),
            # ('word-length', AverageWordLengthExtractor()),
        ])),
    ])

    dumb_feat_pipe = Pipeline([
        ('feature-extraction', FeatureUnion([
            ('word-length', AverageWordLengthExtractor()),
        ])),
    ])
    
    
    clf = LogisticRegression(solver='lbfgs', max_iter=10000)

    #model = pipe.fit(training_data, training_targets)

    #if True:
    #    return

        
    # Create train/test splits
    #X_train, X_test, y_train, y_test = train_test_split(training_data,
    #                                                    training_targets,
    #                                                    test_size=0.2,
    #                                                    random_state=0)

    X = syntax_feat_pipe.fit_transform(training_data)
    #X = dumb_feat_pipe.fit_transform(training_data)
    #X = feat_pipe.fit_transform(training_data)
    #X = training_data
    y = training_targets

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X, y, cv=5, \
                             verbose=1, scoring='roc_auc')    
    print(np.mean(scores), scores)

    #clf = SVC()
    #scores = cross_val_score(clf, X, y, cv=5, \
    #                         verbose=1, scoring='roc_auc')
    #print(np.mean(scores), scores)

    #clf = SVC(kernel='linear')
    #scores = cross_val_score(clf, X, y, cv=5, \
    #                         verbose=1, scoring='roc_auc')
    #print(np.mean(scores), scores)
    
    print('Training and saving full model')
    model = clf.fit(X, y)
    
    with open('full_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()
    
