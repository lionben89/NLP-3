from copy import deepcopy

from crafted_features import calculate_features
from metrics import evaluate_metrics

from preprocess import preprocess
from vectorize import TFIDF, W2VGensim, W2VReduced, W2VGlove, MeanW2V, ConcatW2V
from kfold import KFoldCV
from classifier import LRClassifier, BasicNN, SVMClassifier, LSTMClassifier, TextNumericalInputsClassifier
import numpy as np

VECTOR_SIZE = 50
NUM_OF_WORDS = 10
ID_1 = 313278889
ID_2 = 302680665
N_META_FEATURES = 9
BEST_CLS = TextNumericalInputsClassifier(vector_size=VECTOR_SIZE, n_layers=2, linear_dim=64,
                                         dense_size=64, numeric_feature_size=N_META_FEATURES,
                                         dropout=0.2)  # todo: change cls according to the best one.


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
    return list(y_pred.detach().numpy())


def get_best_model(max_metric, X, y, kf):
    best_model = None
    best_score = 0
    meta_features = calculate_features("trump_train.tsv").to_numpy()

    cls_list = [
        LRClassifier(),
        SVMClassifier(kernel='linear'),
        SVMClassifier(kernel='rbf'),
        SVMClassifier(kernel='poly'),
        BasicNN(input_size=VECTOR_SIZE, n_epochs=3),
        LSTMClassifier(VECTOR_SIZE, 2, 64, dropout=0.2),
        TextNumericalInputsClassifier(vector_size=VECTOR_SIZE, n_layers=2, linear_dim=64,
                                      dense_size=64, numeric_feature_size=meta_features.shape[1], dropout=0.2)
    ]
    for cls in cls_list:
        if isinstance(cls, TextNumericalInputsClassifier):
            X = np.hstack((meta_features, X))
        scores = kf.run_kfold_cv(X, y, cls)
        if scores[max_metric] > best_score:
            best_model = cls
            best_score = scores[max_metric]

    best_model.train(X, y)  # train on all of the data for generalization
    pred = best_model.predict(X)
    print(best_model)
    print(evaluate_metrics(y, pred))

    best_model.save()
    print("final results:")
    print(best_model)
    print(evaluate_metrics(y, pred))


def save_pred_to_file(pred_list):
    """
    Creates the results file. It has a single, space separated line containing only zeros and ones (integers)
     denoting the predicted class (0 for Trump, 1 for a staffer).
     The order of the labels correspond to the tweet order in the testset.
    Args:
        pred_list: predictions list (0 for Trump, 1 for a staffer).
    """
    with open(f'{ID_1}_{ID_2}.txt', 'w') as f:
        for item in pred_list:
            f.write("%s " % item)


if __name__ == '__main__':
    # cls = load_best_model()
    # preds = predict(cls, "trump_test.tsv")
    # save_pred_to_file(preds)
    # train_best_model()

    """ MAIN CODE FIND BEST"""
    vectorize_methods = [TFIDF(VECTOR_SIZE),
                         MeanW2V(W2VGlove(), VECTOR_SIZE),
                         # MeanW2V(W2VGensim(min_count=1, vector_size=VECTOR_SIZE, window=5, sg=1), VECTOR_SIZE),
                         ConcatW2V(W2VGlove(), VECTOR_SIZE, NUM_OF_WORDS),
                         # ConcatW2V(W2VReduced('./gensim_reduced.pkl'), VECTOR_SIZE, NUM_OF_WORDS),
                         ]
    kf = KFoldCV(n_splits=5, shuffle=True)
    ds = preprocess("trump_train.tsv")
    for vectorize in vectorize_methods:
        X, y = vectorize.fit_transform(ds['text'], ds['device'])
        get_best_model("accuracy", X, y, kf)
    
    """ SAVE W2V CODE"""
    # ds = preprocess('trump_train.tsv', train=True)
    # vectorize1 = MeanW2V(W2VGlove(), VECTOR_SIZE)
    # X, y = vectorize1.fit_transform(ds['text'],ds['device'])
    # vectorize1.w2v.save(ds['text'])
    # vectorize1 = MeanW2V(W2VGensim(), VECTOR_SIZE)
    # X, y = vectorize1.fit_transform(ds['text'],ds['device'])
    # vectorize1.w2v.save(ds['text'])

    # for meta-features addition
    # meta_features = calculate_features("trump_train.tsv").to_numpy()
    # X = np.hstack((meta_features, X))
    # cls = TextNumericalInputsClassifier(input_size=VECTOR_SIZE, n_layers=2, linear_dim=64,
    #                                     dense_size=64, numeric_feature_size=meta_features.shape[1], dropout=0.2, n_epochs=5)
