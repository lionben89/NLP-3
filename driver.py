from sklearn.linear_model import LogisticRegression

from preprocess import preprocess
from vectorize import Glove, TFIDF, W2V
from kfold import KFoldCV
from classifier import LRClassifier, BasicNN

if __name__ == '__main__':
    vectore_size = 1200
    # vectorize = TFIDF(vectore_size)
    vectorize = Glove(100,12)
    cls = BasicNN(shape=vectore_size)
    # cls = LRClassifier()
    kf = KFoldCV(n_splits=5, shuffle=True)
    ds = preprocess("trump_train.tsv")
    X, y = vectorize.fit_transform(ds['text'], ds['device'])


    kf.run_kfold_cv(X, y, cls)
