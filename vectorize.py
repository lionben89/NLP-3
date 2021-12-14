from abc import ABC, abstractmethod
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from torchnlp.word_to_vector import GloVe
from sklearn.decomposition import PCA
import numpy as np
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import pickle


class W2VCore(ABC):
    def __init__(self, core):
        super().__init__()
        self.w2v = core

    @abstractmethod
    def get_all_vectors(self):
        pass

    @abstractmethod
    def get_vector(self, token):
        pass

    def text_to_vectors(self, text):
        types = set()
        for sentence in text:
            types |= set(sentence)

        vectors = []
        for t in types:
            vectors.append(self.get_vector(t))

        vectors = np.array(vectors)
        return vectors.reshape(vectors.shape[0], -1)

    def save(self, text, path):
        types = set()
        for sentence in text:
            types |= set(sentence)

        df = pd.DataFrame()
        for t in types:
            df[t] = self.get_vector(t).reshape(-1).tolist()

        df.to_pickle(path)


class W2VGlove(W2VCore):
    def __init__(self, **kargs):
        super().__init__(core=GloVe(**kargs))

    def get_all_vectors(self):
        return self.w2v.vectors.numpy()

    def get_vector(self, token):
        return self.w2v[token].numpy().reshape(1, -1)

    def save(self, text, path='./glove_reduced.pkl'):
        super().save(text, path)


class W2VGensim(W2VCore):
    def __init__(self, **kargs):
        super().__init__(core=KeyedVectors.load_word2vec_format(
            'GoogleNews-vectors-negative300.bin.gz', binary=True))
        # super().__init__(core=Word2Vec(api.load('glove-wiki-gigaword-50'),**kargs))

    def get_all_vectors(self):
        return self.w2v.vectors

    def get_vector(self, token):
        try:
            return self.w2v[token].reshape(1, -1)
        except:
            return np.zeros((1, int(self.w2v.vectors.shape[1])))

    def save(self, text, path='./gensim_reduced.pkl'):
        super().save(text, path)


class W2VReduced(W2VCore):
    def __init__(self, path):
        super().__init__(core=self.load_vectors(path))

    def load_vectors(self, filename):
        return pd.read_pickle(filename)

    def get_all_vectors(self):
        return self.w2v.to_numpy()

    def get_vector(self, token):
        try:
            return self.w2v[token].to_numpy().T.reshape(1, -1)
        except:
            return np.zeros((1, int(self.w2v.shape[0])))


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


class ConcatW2V(Vectorizer):
    def __init__(self, core_w2v, vector_length, number_of_words=10):
        super().__init__(vector_length)
        self.w2v = core_w2v
        self.number_of_words = number_of_words

    def fit(self, text):
        self.pca.fit(self.w2v.text_to_vectors(text))

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
                    embbedings = np.zeros(
                        self.number_of_words * self.vector_length)
                token = sentence[j]
                embbedings[c * self.vector_length:(c + 1) * self.vector_length] = self.pca.transform(
                    self.w2v.get_vector(token))
        return np.array(X), np.array(y)


class MeanW2V(Vectorizer):
    def __init__(self, core_w2v, vector_length):
        super().__init__(vector_length)
        self.w2v = core_w2v

    def fit(self, text):
        vectors = []
        for sentence in text:
            vectors.append(
                np.mean(self.w2v.text_to_vectors([sentence]), axis=0))

        vectors = np.array(vectors)
        self.pca.fit(vectors)

    def transform(self, text, labels):
        X = []
        y = []
        # def set_emb(sentence):
        #     embbedings = np.zeros(len(sentence) * self.vector_length)
        #     for ind, word in enumerate(sentence):
        #         embbedings[ind * self.vector_length: (
        #             ind + 1) * self.vector_length] = self.w2v.get_vector(word)
        #     return embbedings
        # # self.gensim_w2v = gensim.models.Word2Vec(
        #     # text, min_count=1, vector_size=self.vector_length, window=5, sg=1)
        # arr = np.asarray(text.apply(lambda sentence: set_emb(sentence)))
        # for label, emb in zip(labels, arr):
        #     X.append(emb)
        #     y.append(label)

        # self.pca.fit(arr)

        # self.pca.fit(
        #     np.asarray(text.apply(lambda words: np.mean([self.w2v.get_vector(w) for w in words], axis=0)).todense()))
        for i in range(len(text)):
            sentence = text[i]
            vectors = []
            for token in sentence:
                vectors.append(self.w2v.get_vector(token))
            vectors = np.mean(vectors, axis=0)
            X.append(self.pca.transform(vectors).reshape(-1))
            y.append(labels[i])
        return np.array(X), np.array(y)


class TFIDF(Vectorizer):
    def __init__(self, vector_length):
        super().__init__(vector_length)
        self.tfidf_vectorizer = TfidfVectorizer()

    def fit(self, text):
        X = np.asarray(self.tfidf_vectorizer.fit_transform(
            text.apply(lambda X: ' '.join(X))).todense())
        self.pca.fit(X)
        return None

    def transform(self, text, labels):
        X = self.pca.transform(np.array(self.tfidf_vectorizer.transform(
            text.apply(lambda X: ' '.join(X))).todense()))
        y = np.array(labels)
        return X, y
