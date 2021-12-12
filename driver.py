import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import gensim
from gensim.models import word2vec

import nltk
import torch
from preprocess import preprocess





def load_best_model():
    """
    training a classifier from scratch (should be the same classifier and parameters
    returned by load_best_model(). Of course, the final model could be slightly different
    than the one returned by load_best_model(), due to randomization issues.

    This function call training on the data file you received. You could assume it is in the current directory.
    It should trigger the preprocessing and the whole pipeline.

    Returns:
        lm: best performing model that was saved as part of the submission bundle
    """
    ds = preprocess("trump_train.tsv")


    # TODO: complete the pipeline
    pass


def train_best_model():
    """

    Returns:


    """
    pass


def predict(m, fn):
    """

    Args:
        m: the trained model
        fn: the full path to a file in the same format as the test set

    Returns:
        a list of 0s and 1s, corresponding to the lines in the specified file.
    """
    pass
