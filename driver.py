from crafted_features import calculate_features

from preprocess import preprocess
from vectorize import TFIDF, W2VGensim, W2VReduced, W2VGlove, MeanW2V, ConcatW2V
from kfold import KFoldCV
from classifier import LRClassifier, BasicNN, SVMClassifier, LSTMClassifier, TextNumericalInputsClassifier
import numpy as np

if __name__ == '__main__':
    vector_size = 50
    number_of_words = 10
    vectorize = TFIDF(vector_size)
    # vectorize = MeanW2V(W2VGlove(),vector_size)
    # vectorize = MeanW2V(W2VGensim(min_count=1, vector_size=vector_size, window=5, sg=1),vector_size)
    # vectorize = ConcatW2V(W2VGlove(), vector_size, number_of_words)
    # vectorize = ConcatW2V(W2VReduced('./gensim_reduced.pkl'),vector_size,number_of_words)

    # cls = BasicNN(input_size=vector_size)
    # cls = LRClassifier()
    # cls = SVMClassifier(kernel='linear')
    # cls = SVMClassifier(kernel='rbf')
    # cls = SVMClassifier(kernel='poly')
    # cls = LSTMClassifier(vector_size, 2, 64, 64, dropout=0.2)

    kf = KFoldCV(n_splits=5, shuffle=True)
    ds = preprocess("trump_train.tsv")
    X, y = vectorize.fit_transform(ds['text'], ds['device'])
    # vectorize.w2v.save(ds['text'])

    # for meta-features addition
    meta_features = calculate_features("trump_train.tsv").to_numpy()
    X = np.hstack((meta_features, X))
    cls = TextNumericalInputsClassifier(input_size=vector_size, n_layers=2, linear_dim=64,
                                        dense_size=64, numeric_feature_size=meta_features.shape[1], dropout=0.2, n_epochs=5)

    kf.run_kfold_cv(X, y, cls)
