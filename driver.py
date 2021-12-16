from crafted_features import calculate_features
from metrics import evaluate_metrics

from preprocess import preprocess
from vectorize import TFIDF, W2VGensim, W2VReduced, W2VGlove, MeanW2V, ConcatW2V
from kfold import KFoldCV
from classifier import LRClassifier, BasicNN, SVMClassifier, LSTMClassifier, TextNumericalInputsClassifier
import numpy as np

VECTOR_SIZE = 50


def load_best_model():
    """
    Returns:
        returning the best performing model that was saved as part of the submission bundle
    """

    meta_features = calculate_features("trump_train.tsv").to_numpy()
    cls = TextNumericalInputsClassifier(vector_size=VECTOR_SIZE, n_layers=2, linear_dim=64,
                                        dense_size=64, numeric_feature_size=meta_features.shape[1],
                                        dropout=0.2)  # todo: change cls according to the best one.
    cls.load()
    return cls


def train_best_model():
    """
    training a classifier from scratch (should be the same classifier and parameters returned by load_best_model().
     Of course, the final model could be slightly different than the one returned by  load_best_model(),
     due to randomization issues.
     This function call training on the data file you received. You could assume it is in the current directory.
     It should trigger the preprocessing and the whole pipeline.

    Returns:

    """
    number_of_words = 10
    vectorize = TFIDF(VECTOR_SIZE)

    ds = preprocess("trump_train.tsv")
    X, y = vectorize.fit_transform(ds['text'], ds['device'])
    meta_features = calculate_features("trump_train.tsv").to_numpy()
    X = np.hstack((meta_features, X))

    cls = TextNumericalInputsClassifier(vector_size=VECTOR_SIZE, n_layers=2, linear_dim=64,
                                        dense_size=64, numeric_feature_size=meta_features.shape[1],
                                        dropout=0.2)  # todo: change cls according to the best one.

    # kf = KFoldCV(n_splits=5, shuffle=True)
    # scores = kf.run_kfold_cv(X, y, cls)

    cls.train(X, y)  # train on all of the data for generalization

    pred = cls.predict(X)
    print(evaluate_metrics(y, pred))
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

    # todo: complete

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

        best_model.train(X, y)  # train on all of the data for generalization

        pred = best_model.predict(X)
        print(best_model)
        print(evaluate_metrics(y, pred))

        best_model.save()


if __name__ == '__main__':
    number_of_words = 10
    vectorize = TFIDF(VECTOR_SIZE)
    # vectorize = MeanW2V(W2VGlove(),VECTOR_SIZE)
    # vectorize = MeanW2V(W2VGensim(min_count=1, vector_size=VECTOR_SIZE, window=5, sg=1),VECTOR_SIZE)
    # vectorize = ConcatW2V(W2VGlove(), VECTOR_SIZE, number_of_words)
    # vectorize = ConcatW2V(W2VReduced('./gensim_reduced.pkl'),VECTOR_SIZE,number_of_words)

    # cls = BasicNN(input_size=VECTOR_SIZE, n_epochs=3)
    cls = LRClassifier()
    # cls = SVMClassifier(kernel='linear')
    # cls = SVMClassifier(kernel='rbf')
    # cls = SVMClassifier(kernel='poly')
    # cls = LSTMClassifier(VECTOR_SIZE, 2, 64, 64, dropout=0.2)

    kf = KFoldCV(n_splits=5, shuffle=True)
    ds = preprocess("trump_train.tsv")
    X, y = vectorize.fit_transform(ds['text'], ds['device'])
    get_best_model("accuracy", X, y, kf)
    #
    # vectorize.w2v.save(ds['text'])

    # for meta-features addition
    # meta_features = calculate_features("trump_train.tsv").to_numpy()
    # X = np.hstack((meta_features, X))
    # cls = TextNumericalInputsClassifier(input_size=VECTOR_SIZE, n_layers=2, linear_dim=64,
    #                                     dense_size=64, numeric_feature_size=meta_features.shape[1], dropout=0.2, n_epochs=5)

    # cls = cls.load()
    # y_pred = cls.predict(X)
    # print(evaluate_metrics(y, y_pred))
