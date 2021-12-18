from string import punctuation

import nltk

from nltk import word_tokenize
from nltk.corpus import stopwords

from preprocess import load_data, label_encoder
import pandas as pd
import re

nltk.download('stopwords')


def calculate_features(file_name, train=True):
    """
    The method calculates features from the input tweets
    Args:
        file_name: path of the dataset file
        train: If True, then the train mode will load the training set. else, load the test set

    Returns:
        dataset of calculated features as described in the report.
    """
    if train:
        col_list = ['tweet_id', 'user_handle', 'text', 'timestamp', 'device']
        ds = load_data(file_name, col_list)
        ds.dropna(thresh=0, inplace=True)
        ds = label_encoder(ds, None, "device")
        ds.reset_index(drop=True, inplace=True)
    else:
        col_list = ['user_handle', 'text', 'timestamp']
        ds = load_data(file_name, col_list)
        ds.dropna(thresh=0, inplace=True)

    text = ds["text"]
    features_df = pd.DataFrame()
    # A tweet’s number of characters
    features_df["n_chars"] = text.apply(lambda x: len(x))
    # A tweet’s number of words
    features_df["n_words"] = text.apply(lambda x: len(x.split(" ")))
    # Average word length
    features_df["avg_word_len"] = features_df["n_chars"] / features_df["n_words"]
    # A tweet’s number of punctuation characters
    features_df["n_punctuation"] = text.apply(lambda x: len("".join(c for c in x if c in punctuation)))
    # A tweet’s number of unique words
    features_df["n_unique_words"] = text.apply(lambda x: len(set(x)))
    # A tweet’s number of uppercase words (e.g, “HELLO”, “AMAZING”)
    features_df["n_upper_case_words"] = text.apply(lambda x: sum(map(str.isupper, x)))
    # A tweet’s number of stopwords
    features_df["n_stopwords"] = text.apply(lambda x: count_stopwords(x))
    # A tweet’s number of hashtags
    features_df["n_hashtags"] = text.apply(lambda x: len(re.findall(r"(?:^|\s)[＃#]{1}(\w+)", x)))
    # A tweet’s number of mentions
    features_df["n_mentions"] = text.apply(lambda x: len(re.findall(r"(?:^|\s)[＠ @]{1}([^\s#<>[\]|{}]+)", x)))

    return features_df


def count_stopwords(x):
    """
    Counts the number of stopwords within a tweet, using NLTK's stopwords list.
    Args:
        x: tweet as an input

    Returns:
        (int): num of stopwords

    """
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(x)
    stopwords_x = [w for w in word_tokens if w in stop_words]
    return len(stopwords_x)
