from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from torchnlp.word_to_vector import GloVe
from sklearn.decomposition import PCA
import numpy as np
import gensim.downloader


class Vectorizer(ABC):
    def __init__(self, vector_length=200):
        super().__init__()
        self.vector_length = vector_length
        self.pca = PCA(n_components=vector_length)

    @abstractmethod
    def fit(self, text):
        pass

    @abstractmethod
    def transform(self, text, labels):
        pass

    def fit_transform(self, text, labels):
        self.fit(text)
        return self.transform(text, labels)


## not working yet    
class Glove(Vectorizer):
    def __init__(self, vector_length, number_of_words=10):
        super().__init__(vector_length)
        self.w2v_glove = GloVe()
        self.number_of_words = number_of_words

    def fit(self, text):
        self.pca.fit(self.w2v_glove.vectors.numpy()) ## fit on all or just on trainset or per sentence

    def save(self):
        return None

    def transform(self, text, labels):
        X = []
        y = []
        for i in range(len(text)):
            sentence = text[i]
            embbedings = np.zeros(self.number_of_words * self.vector_length)
            for j in range(len(sentence)):
                c = j % self.number_of_words
                if (c == 0 and j > 0):
                    X.append(embbedings)
                    y.append(labels[i])
                    embbedings = np.zeros(self.number_of_words * self.vector_length)
                token = sentence[j]
                embbedings[c * self.vector_length:(c + 1) * self.vector_length] = self.pca.transform(
                    self.w2v_glove[token].numpy().reshape(1,-1))
        return np.array(X), np.array(y)


class TFIDF(Vectorizer):
    def __init__(self, vector_length):
        super().__init__(vector_length)
        self.tfidf_vectorizer = TfidfVectorizer()

    def fit(self, text):
        X = np.asarray(self.tfidf_vectorizer.fit_transform(text.apply(lambda X: ' '.join(X))).todense())
        self.pca.fit(X)
        return None

    def transform(self, text, labels):
        X = self.pca.transform(np.array(self.tfidf_vectorizer.transform(text.apply(lambda X: ' '.join(X))).todense()))
        y = np.array(labels)
        return X, y

## not working!
class W2V(Vectorizer):
    def __init__(self, vector_length):
        super().__init__(vector_length)
        self.gensim_w2v = None

    def fit(self, text):

        return None

    def transform(self, text, labels):
        def set_emb(sentence):
            embbedings = np.zeros(len(sentence) * self.vector_length)
            for ind, word in enumerate(sentence):
                embbedings[ind * self.vector_length: (ind + 1) * self.vector_length] = self.gensim_w2v.wv[word]
            return embbedings

        X = []
        y = []
        self.gensim_w2v = gensim.models.Word2Vec(text, min_count=1, vector_size=self.vector_length, window=5, sg=1)
        arr = np.asarray(text.apply(lambda sentence: set_emb(sentence)))
        for label, emb in zip(labels, arr):
            X.append(emb)
            y.append(label)

        self.pca.fit(arr)

        self.pca.fit(
            np.asarray(text.apply(lambda words: np.mean([self.gensim_w2v.wv[w] for w in words], axis=0)).todense()))
        return X, y

    def save(self, text):
        filename = "embeddings.csv"

        pass
