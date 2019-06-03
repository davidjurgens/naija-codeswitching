import csv
import sys
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.utils.multiclass import unique_labels


import numpy as np
import matplotlib.pyplot as plt
import scipy

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized tribal affiliation confusion matrix")
    else:
        print('Tribal Affiliation Confusion Matrix')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__ == "__main__":

  X = []
  y = []

  with open('/Users/sghosh/Development/NigeriaMediaCorpus/CODE/held_out_author.tsv', 'r') as f:
    tsv_reader = csv.reader(f)
    next(tsv_reader, None)

    for annotation in tsv_reader:
      X.append(annotation[0].lower())
      y.append(annotation[1].strip())

  with open('names_classifier.pkl', 'rb') as f:
    model, labels, dummy = pickle.load(f)

  print(labels)
  y_true = []
  for annotation in y:
    y_true.append(labels.index(annotation.lower()))

  y_majority = dummy.predict(X)
  y_random = np.random.randint(0, 4, len(y_true))
  
  predictions = model.predict(X)

  for counter, value in enumerate(X):
    print("{} {}\n".format(value, labels[predictions[counter]]))

  print(f1_score(y_true, predictions, average='macro'))
  print(f1_score(y_true, y_majority, average='macro'))
  print(f1_score(y_true, y_random, average='macro'))
  plot_confusion_matrix(y_true, predictions, labels, normalize=True)
  plt.show()