from sklearn.linear_model import LogisticRegression

from preprocess import preprocess
from vectorize import TFIDF, W2VGensim, W2VReduced, W2VGlove, MeanW2V, ConcatW2V
from kfold import KFoldCV
from classifier import LRClassifier, BasicNN, SVMClassifier, LSTMClassifier

if __name__ == '__main__':
    vectore_size = 50
    number_of_words=12
    # vectorize = TFIDF(vectore_size)
    # vectorize = MeanW2V(W2VGlove(),vectore_size)
    # vectorize = MeanW2V(W2VGensim(min_count=1, vector_size=vectore_size, window=5, sg=1),vectore_size)
    # vectorize = ConcatW2V(W2VGlove(),vectore_size,number_of_words)
    vectorize = ConcatW2V(W2VReduced('./gensim_reduced.pkl'),vectore_size,number_of_words)
    
    # cls = BasicNN(shape=vectore_size*number_of_words)
    # cls = LRClassifier()
    # cls = SVMClassifier(kernel='linear')
    # cls = SVMClassifier(kernel='rbf')
    # cls = SVMClassifier(kernel='poly')
    cls = LSTMClassifier(vectore_size,12,64,dropout=0.2)
    
    kf = KFoldCV(n_splits=5, shuffle=True)
    ds = preprocess("trump_train.tsv")
    X, y = vectorize.fit_transform(ds['text'], ds['device'])
    # vectorize.w2v.save(ds['text'])

    kf.run_kfold_cv(X, y, cls)
