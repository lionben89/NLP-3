from string import punctuation

import nltk

from nltk import word_tokenize
from nltk.corpus import stopwords

from preprocess import load_data, label_encoder
import pandas as pd
import re

nltk.download('stopwords')


def calculate_features(file_name, train=True):
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
    features_df["n_chars"] = text.apply(lambda x: len(x))
    features_df["n_words"] = text.apply(lambda x: len(x.split(" ")))
    features_df["avg_word_len"] = features_df["n_chars"] / features_df["n_words"]
    features_df["n_punctuation"] = text.apply(lambda x: len("".join(c for c in x if c in punctuation)))
    features_df["n_unique_words"] = text.apply(lambda x: len(set(x)))
    features_df["n_upper_case_words"] = text.apply(lambda x: sum(map(str.isupper, x)))
    features_df["n_stopwords"] = text.apply(lambda x: count_stopwords(x))
    features_df["n_hashtags"] = text.apply(lambda x: len(re.findall(r"(?:^|\s)[＃#]{1}(\w+)", x)))
    features_df["n_mentions"] = text.apply(lambda x: len(re.findall(r"(?:^|\s)[＠ @]{1}([^\s#<>[\]|{}]+)", x)))

    return features_df


def count_stopwords(x):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(x)
    stopwords_x = [w for w in word_tokens if w in stop_words]
    return len(stopwords_x)
