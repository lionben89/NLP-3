from copy import deepcopy

from crafted_features import calculate_features
from metrics import evaluate_metrics

from preprocess import preprocess
from vectorize import TFIDF, W2VGensim, W2VReduced, W2VGlove, MeanW2V, ConcatW2V
from kfold import KFoldCV
from classifier import LRClassifier, BasicNN, SVMClassifier, LSTMClassifier, TextNumericalInputsClassifier
from visualize import plot_all
import numpy as np
import pickle
import pandas as pd

# PARAMS
VECTOR_SIZE = 50
CONCAT_VECTOR_SIZE = 30
NUM_OF_WORDS = 14
ID_1 = 313278889
ID_2 = 302680665
N_META_FEATURES = 9
BEST_CLS = SVMClassifier(kernel='rbf')


def load_best_model():
    """
    Returns:
        returning the best performing model that was saved as part of the submission bundle
    """
    cls = deepcopy(BEST_CLS)
    cls = cls.load()
    return cls


def train_best_model():
    """
    training a classifier from scratch (should be the same classifier and parameters returned by load_best_model().
    This function call training on the data file "trump_train.csv", assuming it is in the current directory.
    It triggers the preprocessing and the whole pipeline.

    Returns:
            cls: trained classifier
    """
    vectorize = TFIDF(VECTOR_SIZE)
    cls = deepcopy(BEST_CLS)

    ds = preprocess("trump_train.tsv")
    X, y = vectorize.fit_transform(ds['text'], ds['device'])

    if isinstance(cls, TextNumericalInputsClassifier):
        meta_features = calculate_features("trump_train.tsv").to_numpy()
        X = np.hstack((meta_features, X))

    cls.train(X, y)  # train on all of the data for generalization

    return cls


def predict(m, fn):
    """
    Args:
        m: the trained model
        fn: the full path to a file in the same format as the test set

    Returns:
        list: a list of 0s and 1s, corresponding to the lines in the specified file.
    """
    vectorize = TFIDF(VECTOR_SIZE)

    ds = preprocess(fn, train=False)
    X, _ = vectorize.fit_transform(ds['text'], [])

    if isinstance(m, TextNumericalInputsClassifier):
        meta_features = calculate_features(fn, train=False).to_numpy()
        X = np.hstack((meta_features, X))

    y_pred = m.predict(X)
    if isinstance(y_pred, np.ndarray):
        lst = list(y_pred)
    else:
        lst = list(y_pred.detach().numpy())
    return lst


def get_best_model(max_metric, fn):
    """
    The function compares performances across all algorithms and vectorization methods
    Args:
        max_metric: metric to compare by (accuracy, precision, recall, auc or f1)
        fn: the full path to a file in the same format as "trump_train.tsv"

    Returns:
        data: a list with all scores for each algorithm & vectorization method
    """
    best_model = None
    best_vectorize = None
    best_score = 0
    ds = preprocess(fn)
    meta_features = calculate_features(fn).to_numpy()
    data = []
    vectorize_methods = [
        TFIDF(VECTOR_SIZE),
        ConcatW2V(W2VReduced('./glove_reduced'),
                  CONCAT_VECTOR_SIZE, NUM_OF_WORDS),
        ConcatW2V(W2VReduced('./gensim_reduced'),
                  CONCAT_VECTOR_SIZE, NUM_OF_WORDS),
        MeanW2V(W2VReduced('./glove_reduced'), VECTOR_SIZE),
        MeanW2V(W2VReduced('./gensim_reduced'), VECTOR_SIZE),

        # ConcatW2V(W2VGlove(), CONCAT_VECTOR_SIZE, NUM_OF_WORDS),
        # ConcatW2V(W2VGensim(), CONCAT_VECTOR_SIZE, NUM_OF_WORDS),
    ]

    kf = KFoldCV(n_splits=5, shuffle=True)

    for vectorize in vectorize_methods:
        if isinstance(vectorize, ConcatW2V):
            num_of_words = NUM_OF_WORDS
            vector_size = CONCAT_VECTOR_SIZE
        else:
            num_of_words = 1
            vector_size = VECTOR_SIZE

        cls_list = [
            LRClassifier(),
            SVMClassifier(kernel='linear'),
            SVMClassifier(kernel='rbf'),
            # # SVMClassifier(kernel='poly'),
            BasicNN(input_size=vector_size*num_of_words, n_epochs=10),
            LSTMClassifier(vector_size, 2, 32, dropout=0.1, n_epochs=20),
            TextNumericalInputsClassifier(vector_size=vector_size, n_layers=2, linear_dim=32, n_epochs=20,
                                          dense_size=16, numeric_feature_size=meta_features.shape[1], dropout=0.1)
        ]

        if (isinstance(vectorize, ConcatW2V) and isinstance(cls, TextNumericalInputsClassifier)):
            X, y, m = vectorize.fit_transform(
                ds['text'], ds['device'], meta_features)
        else:
            X, y = vectorize.fit_transform(ds['text'], ds['device'])

        for cls in cls_list:
            if isinstance(cls, TextNumericalInputsClassifier):
                if (isinstance(vectorize, ConcatW2V)):
                    X = np.hstack((m, X))
                else:
                    X = np.hstack((meta_features, X))

            print("running {}_{}".format(cls.to_string(), vectorize.to_string()))
            scores, fpr, tpr, cls_trained = kf.run_kfold_cv(X, y, cls)
            print("{}_{} results are: {}".format(
                cls.to_string(), vectorize.to_string(), scores))
            data.append({"classifier": cls_trained, "scores": scores,
                        "vectorize": vectorize, "fpr": fpr, "tpr": tpr})
            if scores[max_metric] > best_score:
                best_model = cls
                best_vectorize = vectorize
                best_score = scores[max_metric]

    print("final results:")
    best_model.train(X, y)  # train on all of the data for generalization
    pred = best_model.predict(X)
    print("best_model is: {}_{}".format(
        best_model.to_string(), best_vectorize.to_string()))
    print(evaluate_metrics(y, pred))
    best_model.save()
    pickle.dump(data, open("get_best_model_results.pkl", 'wb'))
    return data


def load_data():
    data = pickle.load(open("get_best_model_results.pkl", 'rb'))
    return data


def save_pred_to_file(pred_list):
    """
    Creates the results file. It has a single, space separated line containing only zeros and ones (integers)
     denoting the predicted class (0 for Trump, 1 for a staffer).
     The order of the labels correspond to the tweet order in the test set.
    Args:
        pred_list: predictions list (0 for Trump, 1 for a staffer).
    """
    with open(f'{ID_1}_{ID_2}.txt', 'w') as f:
        for item in pred_list:
            f.write("%s " % item)

def trump_test(m, fn):
    """
    Args:
        m: the trained model
        fn: the full path to a file in the same format as the test set

    Returns:
        list: a list of 0s and 1s, corresponding to the lines in the specified file.
    """
    vectorize = TFIDF(VECTOR_SIZE)

    ds = preprocess(fn, train=True)
    X, _ = vectorize.fit_transform(ds['text'], ds['device'])

    if isinstance(m, TextNumericalInputsClassifier):
        meta_features = calculate_features(fn, train=False).to_numpy()
        X = np.hstack((meta_features, X))

    y_pred = m.predict(X)
    mask = (ds['timestamp'] < pd.Timestamp(year=2016, month=11, day=8)) & (ds['timestamp'] > pd.Timestamp(year=2016, month=10, day=8))
    print("true:")
    print(ds['device'][mask].to_list())
    print("prediction:")
    print(list(y_pred[mask]))
    print(evaluate_metrics(ds['device'][mask].to_list(),list(y_pred[mask])))

if __name__ == '__main__':
    cls = train_best_model()
    cls.save()
    cls = load_best_model()
    preds = predict(cls, "trump_test.tsv")
    save_pred_to_file(preds)

    """ MAIN CODE FIND BEST"""
    # data = get_best_model("accuracy", "trump_train.tsv")
    # data = load_data()
    # plot_all(data)

    """ SAVE W2V CODE"""
    # ds = preprocess('trump_train.tsv', train=True)
    # vectorize1 = MeanW2V(W2VGlove(), CONCAT_VECTOR_SIZE)
    # X, y = vectorize1.fit_transform(ds['text'],ds['device'])
    # vectorize1.w2v.save(ds['text'])
    # vectorize1 = MeanW2V(W2VGensim(), CONCAT_VECTOR_SIZE)
    # X, y = vectorize1.fit_transform(ds['text'],ds['device'])
    # vectorize1.w2v.save(ds['text'])

    # for meta-features addition
    # meta_features = calculate_features("trump_train.tsv").to_numpy()
    # X = np.hstack((meta_features, X))
    # cls = TextNumericalInputsClassifier(input_size=VECTOR_SIZE, n_layers=2, linear_dim=64,
    #                                     dense_size=64, numeric_feature_size=meta_features.shape[1], dropout=0.2, n_epochs=5)

    """ TRUMP TEST"""
    m = train_best_model()
    pred = trump_test(m,'trump_train.tsv')
    print(pred)