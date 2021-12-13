import pandas as pd
from sklearn import preprocessing
import nltk 
nltk.download('punkt')

dataset_structure = None

TIMESTAMP_FEATURES = {
    "timestamp": True,
    "day_of_week": True,
    "day_of_month": True,
    "month": True,
    "hour": True,
    "minute": True,
}

def preprocess(filename):
    """[summary]

    Args:
        filename ([type]): [description]

    Returns:
        [type]: [description]
    """

    dataset_structure = [{"name": "tweet_id", "func": empty_func},
                         {"name": "user_handle", "func": dummy_encoder},
                         {"name": "text", "func": text_preprocess},
                         {"name": "timestamp", "func": timestamp_preprocess},
                         {"name": "device", "func": label_encoder}]
    column_names = list(map(lambda col_s: col_s["name"],dataset_structure))
    ds = load_data(filename, column_names)
    ds.dropna(thresh=0, inplace=True)
    for i in range(len(dataset_structure)):
        column_structure = dataset_structure[i]
        ds = column_structure["func"](ds, i, column_structure["name"])
        ds.reset_index()

    return ds


def load_data(filename, column_names):
    """[summary]

    Args:
        filename ([type]): [description]

    Returns:
        [type]: [description]
    """
    ds = pd.read_table(filename, names=column_names)
    return ds


def empty_func(ds, column, name):
    return ds


def dummy_encoder(ds, column, name):
    """[summary]

    Args:
        ds ([type]): [description]
        column ([type]): [description]
        name ([type]): [description]

    Returns:
        [type]: [description]
    """
    dummies = pd.get_dummies(ds[name], prefix=name)
    ds = ds.drop(columns=[name])
    ds = pd.concat([ds,dummies],axis=1)
    return ds


def remove_whitespace(text):
    return " ".join(text.split())


def remove_punct(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    lst = tokenizer.tokenize(' '.join(text))
    return lst


def text_preprocess(ds, column, name):
    text = ds[name]
    text = text.str.lower()
    text = text.apply(remove_whitespace)
    text = text.apply(lambda X: nltk.word_tokenize(X))
    text = text.apply(lambda X: remove_punct(X))
    ds[name] = text
    return ds


def timestamp_preprocess(ds, column, name):
    """[summary]

    Args:
        ds ([type]): [description]
        column ([type]): [description]
        name ([type]): [description]

    Returns:
        [type]: [description]
    """
    ts = pd.to_datetime(ds[name])
    for feature in TIMESTAMP_FEATURES.keys():
        if TIMESTAMP_FEATURES[feature] is not None:
            if feature == "timestamp":
                ds[feature] = ts
            elif feature == "day_of_week":
                ds[feature] = ts.apply(lambda X: X.day_of_week)
            elif feature == "day_of_month":
                ds[feature] = ts.apply(lambda X: X.day)
            elif feature == "month":
                ds[feature] = ts.apply(lambda X: X.month)
            elif feature == "hour":
                ds[feature] = ts.apply(lambda X: X.hour)
            elif feature == "minute":
                ds[feature] = ts.apply(lambda X: X.minute)
    return ds


def label_encoder(ds, column, name):
    """[summary]

    Args:
        ds ([type]): [description]
        column ([type]): [description]
        name ([type]): [description]

    Returns:
        [type]: [description]
    """
    alowed_labels = ["android", "iphone"]
    ds = ds[ds[name].isin(alowed_labels)]
    le = preprocessing.LabelEncoder()
    le.fit(ds[name])
    ds[name] = le.transform(ds[name])
    ## iphone 0 , android 1
    return ds


if __name__ == '__main__':
    preprocess("trump_train.tsv")
