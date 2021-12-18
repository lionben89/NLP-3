from copy import deepcopy

from crafted_features import calculate_features
from metrics import evaluate_metrics

from preprocess import preprocess
from vectorize import TFIDF, W2VGensim, W2VReduced, W2VGlove, MeanW2V, ConcatW2V
from kfold import KFoldCV
from classifier import LRClassifier, BasicNN, SVMClassifier, LSTMClassifier, TextNumericalInputsClassifier
from visualize import plot_all
import numpy as np

VECTOR_SIZE = 50
CONCAT_VECTOR_SIZE = 20
NUM_OF_WORDS = 10
ID_1 = 313278889
ID_2 = 302680665
N_META_FEATURES = 9
BEST_CLS = TextNumericalInputsClassifier(vector_size=VECTOR_SIZE, n_layers=2, linear_dim=64,
                                         dense_size=64, numeric_feature_size=N_META_FEATURES,
                                         dropout=0.2)  # todo: change cls according to the best one.
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
    best_model = None
    best_vectorize = None
    best_score = 0
    ds = preprocess(fn)
    meta_features = calculate_features(fn).to_numpy()
    data = []
    vectorize_methods = [
        TFIDF(VECTOR_SIZE),
        MeanW2V(W2VReduced('./glove_reduced_50.pkl'), VECTOR_SIZE),
        MeanW2V(W2VReduced('./gensim_reduced_50.pkl'), VECTOR_SIZE),
        ConcatW2V(W2VReduced('./glove_reduced_20.pkl'),
                  CONCAT_VECTOR_SIZE, NUM_OF_WORDS),
        ConcatW2V(W2VReduced('./gensim_reduced_20.pkl'),
                  CONCAT_VECTOR_SIZE, NUM_OF_WORDS),

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
            # SVMClassifier(kernel='poly'),
            BasicNN(input_size=vector_size * num_of_words, n_epochs=3),
            LSTMClassifier(vector_size, 2, 64, dropout=0.2),
            TextNumericalInputsClassifier(vector_size=vector_size, n_layers=2, linear_dim=64,
                                          dense_size=64, numeric_feature_size=meta_features.shape[1], dropout=0.2)
        ]

        if (isinstance(vectorize, ConcatW2V) and isinstance(cls, TextNumericalInputsClassifier)):
            X, y, m = vectorize.fit_transform(ds['text'], ds['device'], meta_features)
        else:
            X, y = vectorize.fit_transform(ds['text'], ds['device'])

        for cls in cls_list:
            if isinstance(cls, TextNumericalInputsClassifier):
                if (isinstance(vectorize, ConcatW2V)):
                    X = np.hstack((m, X))
                else:
                    X = np.hstack((meta_features, X))

            print("running {}_{}".format(cls.to_string(), vectorize.to_string()))
            scores, fpr, tpr = kf.run_kfold_cv(X, y, cls)
            print("{}_{} results are: {}".format(
                cls.to_string(), vectorize.to_string(), scores))
            data.append({"classifier": cls, "scores": scores,
                         "vectorize": vectorize, "fpr": fpr, "tpr": tpr})
            if scores[max_metric] > best_score:
                best_model = cls
                best_vectorize = vectorize
                best_score = scores[max_metric]

    print("final results:")
    best_model.train(X, y)  # train on all of the data for generalization
    pred = best_model.predict(X)
    print("best_model is: {}_{}".format(best_model.to_string(), best_vectorize.to_string()))
    print(evaluate_metrics(y, pred))
    best_model.save()
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


if __name__ == '__main__':
    cls = train_best_model()
    cls.save()
    cls = load_best_model()
    preds = predict(cls, "trump_test.tsv")
    save_pred_to_file(preds)

    ds = preprocess('trump_train.tsv', train=True)
    vectorize = TFIDF(VECTOR_SIZE)
    X, y = vectorize.fit_transform(ds['text'], ds['device'])
    preds = cls.predict(X)

    import pandas as pd
    import matplotlib.pyplot as plt

    ds["datetime"] = pd.to_datetime(ds["timestamp"])
    ds["target"] = preds

    fig, axs = plt.subplots(figsize=(12, 4))
    ds.groupby(ds["datetime"].dt.hour)["target"].mean().plot(
        kind='bar', rot=0, ax=axs)
    plt.show()



    """ MAIN CODE FIND BEST"""
    # data = get_best_model("accuracy", "trump_train.tsv")
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
