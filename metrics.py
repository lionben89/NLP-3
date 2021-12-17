from sklearn import metrics


def evaluate_metrics(y_test, y_pred):
    scores = {}
    scores['accuracy'] = metrics.accuracy_score(y_test, y_pred)
    scores['precision'] = metrics.precision_score(y_test, y_pred, average='macro')
    scores['recall'] = metrics.recall_score(y_test, y_pred, average='macro')
    scores['auc'] = metrics.roc_auc_score(y_test, y_pred)
    scores['f1'] = metrics.f1_score(y_test, y_pred)

    return scores

def evaluate_roc(y_test,y_prob):
    fpr, tpr, th = metrics.roc_curve(y_test, y_prob, pos_label=1)
    return fpr,tpr    
