from sklearn import metrics


def evaluate_metrics(y_test, y_pred):
    scores = {}
    scores['accuracy'] = metrics.accuracy_score(y_test, y_pred)
    scores['precision'] = metrics.precision_score(y_test, y_pred, average='macro')
    scores['recall'] = metrics.recall_score(y_test, y_pred, average='macro')
    scores['auc'] = metrics.roc_auc_score(y_test, y_pred)
    scores['f1'] = metrics.f1_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    scores['fpr'] = fpr
    scores['tpr'] = tpr

    return scores
