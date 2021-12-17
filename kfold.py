from sklearn.model_selection import KFold
from collections import Counter
from copy import deepcopy
from metrics import evaluate_metrics, evaluate_roc


class KFoldCV():
    def __init__(self, n_splits=5, shuffle=True):
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle)

    def run_kfold_cv(self, X, y, cls):
        scores = {'accuracy': 0, 'precision': 0, 'recall': 0, 'auc': 0, 'f1': 0}
        i = 0
        for train_index, test_index in self.kf.split(X, y):
            i += 1
            # print("Fold number:{} strat".format(i))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            temp_cls = deepcopy(cls)
            temp_cls.train(X_train, y_train)
            y_pred = temp_cls.predict(X_test)
            # print("Fold number:{} finished".format(i))
            fold_scores = evaluate_metrics(y_test, y_pred)
            for key in scores:
                scores[key] += fold_scores[key]

        # get average scores
        for key in scores:
            scores[key] /= self.n_splits
        
        temp_cls = deepcopy(cls)
        temp_cls.train(X,y)
        y_prob = temp_cls.predict_proba(X)
        fpr, tpr = evaluate_roc(y, y_prob)

        return scores, fpr, tpr
