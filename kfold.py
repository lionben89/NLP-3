from sklearn.model_selection import KFold
from collections import Counter

from metrics import evaluate_metrics


class KFoldCV():
    def __init__(self, n_splits=5, shuffle=True):
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle)

    def run_kfold_cv(self, X, y, cls):
        scores = {'accuracy': 0, "fpr": 0, 'tpr': 0, 'precision': 0, 'recall': 0, 'auc': 0, 'f1': 0}

        for train_index, test_index in self.kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            cls.train(X_train, y_train)
            y_pred = cls.predict(X_test)
            fold_scores = evaluate_metrics(y_test, y_pred)
            for key in scores:
                scores[key] += fold_scores[key]

        # get average scores
        for key in scores:
            scores[key] /= self.n_splits

        print(scores)  # todo: delete
