# FIRST: RENAME THIS FILE TO sentiment_utils.py 

# YOUR NAMES HERE:


"""
Felix Muzny
CS 4/6120
Homework 4
Fall 2023

Utility functions for HW 4, to be imported into the corresponding notebook(s).

Add any functions to this file that you think will be useful to you in multiple notebooks.
"""
# fancy data structures
from collections import defaultdict, Counter
# for tokenizing and precision, recall, f_measure, and accuracy functions
import nltk
from nltk.metrics import ConfusionMatrix
# for plotting
import matplotlib.pyplot as plt
# so that we can indicate a function in a type hint
from typing import Callable
import numpy as np
nltk.download('punkt')

def generate_tuples_from_file(training_file_path: str) -> list:
    """
    Generates data from file formated like:

    tokenized text from file: [[word1, word2, ...], [word1, word2, ...], ...]
    labels: [0, 1, 0, 1, ...]
    
    Parameters:
        training_file_path - str path to file to read in
    Return:
        a list of lists of tokens and a list of int labels
    """
    # PROVIDED
    f = open(training_file_path, "r", encoding="utf8")
    X = []
    y = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.strip().split("\t")
        if len(dataInReview) != 3:
            continue
        else:
            t = tuple(dataInReview)
            if (not t[2] == '0') and (not t[2] == '1'):
                print("WARNING")
                continue
            X.append(nltk.word_tokenize(t[1]))
            y.append(int(t[2]))
    f.close()  
    return X, y


"""
NOTE: for all of the following functions, we have prodived the function signature and docstring, *that we used*, as a guide.
You are welcome to implement these functions as they are, change their function signatures as needed, or not use them at all.
Make sure that you properly update any docstrings as needed.
"""


def get_prfa(dev_y: list, preds: list, verbose=False) -> tuple:
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.
    Args:
        dev_y: list of labels
        preds: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    #TODO: implement this function
    cm = ConfusionMatrix(dev_y, preds)
    p = cm.precision(1)
    r = cm.recall(1)
    f1 = (2*p*r)/(p+r)
    a = nltk.accuracy(dev_y, preds)
    if verbose:
        print(f"Precision: {p:.2f}")
        print(f"Recall: {r:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Accuracy: {a:.2f}")
    return (p,r,f1,a)

def create_training_graph(train_feats: list, dev_feats: list, kind: str, savepath: str = None, verbose: bool = False) -> None:
    """
    Create a graph of the classifier's performance on the dev set as a function of the amount of training data.
    Args:
        metrics_fun: a function that takes in training data and dev data and returns a tuple of metrics
        train_feats: a list of training data in the format [(feats, label), ...]
        dev_feats: a list of dev data in the format [(feats, label), ...]
        kind: the kind of model being used (will go in the title)
        savepath: the path to save the graph to (if None, the graph will not be saved)
        verbose: whether to print the metrics
    """
    #TODO: implement this function
    plt.plot(train_feats, dev_feats[0], label='Precision', color="Blue")
    plt.plot(train_feats, dev_feats[1], label='Recall',  color='Red')
    plt.plot(train_feats, dev_feats[2], label='F1 Score',  color='Green')
    plt.plot(train_feats, dev_feats[3], label='Accuracy',  color='Yellow')

    # Set the title and labels for the x and y axes
    plt.title('Performance Metrics of {} Over % of train data'.format(kind))
    plt.xlabel('% of Taining data')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()

    if savepath is not None: 
        plt.savefig(savepath)
    



def create_index(all_train_data_X: list) -> list:
    """
    Given the training data, create a list of all the words in the training data.
    Args:
        all_train_data_X: a list of all the training data in the format [[word1, word2, ...], ...]
    Returns:
        vocab: a list of all the unique words in the training data
    """
    # figure out what our vocab is and what words correspond to what indices
    #TODO: implement this function
    vocab = [word for lst in all_train_data_X for word in lst]
    return list(set(vocab))


def featurize(vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
    """
    Create vectorized BoW representations of the given data.
    Args:
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out progress
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    # using a Counter is essential to having this not take forever
    #TODO: implement this function
    bow_ve = np.zeros((len(data_to_be_featurized_X), len(vocab)), dtype=int)
    for i, ind in enumerate(data_to_be_featurized_X):
        count = Counter(ind)
        for j, w in enumerate(vocab):
            if binary:
                bow_ve[i][j] = int(w in count)
            else:
                bow_ve[i][j] = count[w]
                          
    if verbose:
        print(bow_ve)
    return bow_ve
