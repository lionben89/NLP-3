from abc import ABC, abstractmethod
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from torchnlp.word_to_vector import GloVe
from sklearn.decomposition import PCA
import numpy as np
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from zipfile import ZipFile
import os

""" This Class is an abstract class that represents the Word to Vec interface and abstract methods"""
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
        """This function gets text and returns a list of vectors from all the types (each unique word) in the text

        Args:
            text ([list(list(string))]): list of lists of tokens

        Returns:
            [list(np.array)]: [list of the vectors of the typs]
        """
        types = set()
        for sentence in text:
            types |= set(sentence)

        vectors = []
        for t in types:
            vectors.append(self.get_vector(t))

        vectors = np.array(vectors)
        return vectors.reshape(vectors.shape[0], -1)

    def save(self, text, path):
        """This function saves the subset of vectors of the types in the text into a zip file

        Args:
            text ([list(list(string))]): list of lists of tokens
            path ([string]):path to save
        """
        types = set()
        for sentence in text:
            types |= set(sentence)

        df = pd.DataFrame()
        for t in types:
            df[t] = self.get_vector(t).reshape(-1).tolist()

        df.to_pickle(path+'.pkl')
        zip_file = ZipFile("{}.zip".format(path), "w")
        zip_file.write("{}.pkl".format(path))
        zip_file.close()
        os.remove(path+'.pkl')
    
    @abstractmethod
    def to_string(self):
        pass

""" This Class implenents the W2VCore class using the pretraind glove embeddings"""
class W2VGlove(W2VCore):
    def __init__(self, **kargs):
        super().__init__(core=GloVe(**kargs))

    def get_all_vectors(self):
        """This function returns all the vectors

        Returns:
            [list(np.array)]: [list of all the vectors]
        """
        return self.w2v.vectors.numpy()

    def get_vector(self, token):
        """This function returns the vector of the token

        Args:
            token ([string]): token to be vectorized

        Returns:
            [np.array]: vector
        """
        return self.w2v[token].numpy().reshape(1, -1)

    def save(self, text, path="./glove_reduced"):
        super().save(text, path)
    
    def to_string(self):
        return "W2V_glove"

""" This Class implenents the W2VCore class using the pretraind gensim google embeddings"""
class W2VGensim(W2VCore):
    def __init__(self, **kargs):
        super().__init__(core=KeyedVectors.load_word2vec_format(
            'GoogleNews-vectors-negative300.bin.gz', binary=True))
        # super().__init__(core=Word2Vec(api.load('glove-wiki-gigaword-50'),**kargs))

    def get_all_vectors(self):
        """This function returns all the vectors

        Returns:
            [list(np.array)]: [list of all the vectors]
        """
        return self.w2v.vectors

    def get_vector(self, token):
        """This function returns the vector of the token, vector of 0 if does not exist

        Args:
            token ([string]): token to be vectorized

        Returns:
            [np.array]: vector
        """
        try:
            return self.w2v[token].reshape(1, -1)
        except:
            return np.zeros((1, int(self.w2v.vectors.shape[1])))

    def save(self, text, path='./gensim_reduced'):
        super().save(text, path)
        
    def to_string(self):
        return "W2V_gensim"

""" This Class implenents the W2VCore class using a pretraind partial embeddings loadded from a zip file"""
class W2VReduced(W2VCore):
    def __init__(self, path):
        super().__init__(core=self.load_vectors(path))
        self.path = path

    def load_vectors(self, filename):
        """This function gets a zip file and load this embeddings

        Args:
            filename ([string]): [zip file that has pickle file of dataframe]

        Returns:
            [dataframe]: [the loaded embeddings]
        """
        with ZipFile(filename+'.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
        return pd.read_pickle(filename+'.pkl')

    def get_all_vectors(self):
        """This function returns all the vectors

        Returns:
            [list(np.array)]: [list of all the vectors]
        """        
        return self.w2v.to_numpy()

    def get_vector(self, token):
        """This function returns the vector of the token, vector of 0 if does not exist

        Args:
            token ([string]): token to be vectorized

        Returns:
            [np.array]: vector
        """
        try:
            return self.w2v[token].to_numpy().T.reshape(1, -1)
        except:
            return np.zeros((1, int(self.w2v.shape[0])))

    def to_string(self):
        return "W2V_reduced_{}".format(self.path)

""" This class represents an interface to vectorizer"""
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

    def fit_transform(self, text, labels, meta_features=None):
        """This function gets X, y and meta features fit the vectorize model and transforms the dataset

        Args:
            text ([list(list(string))]): list of lists of tokens
            labels ([list(integer)]):list of labels
            meta_features ([type], optional): list of meta features. Defaults to None.

        Returns:
            [dataframe]: [vectorize dataset]
        """
        self.fit(text)
        return self.transform(text, labels, meta_features)

""" This class implements the Vectorize class using a concatination of vectors for each sample"""
class ConcatW2V(Vectorizer):
    def __init__(self, core_w2v, vector_length, number_of_words=10):
        """Initiates the class

        Args:
            core_w2v (W2VCore): glove, reduced or gensim
            vector_length (integer): size of the reduced wanted vecor length
            number_of_words (int, optional): number of words in a sample. Defaults to 10.
        """
        super().__init__(vector_length)
        self.w2v = core_w2v
        self.number_of_words = number_of_words

    def fit(self, text):
        """This function fits the PCA

        Args:
            text ([list(list(vector))]): list of lists of vector tokens
        """
        self.pca.fit(self.w2v.text_to_vectors(text))

    def transform(self, text, labels,meta_features):
        """This function transform the dataset into vectorize manner with 
            concatination of reduced size vectors

        Args:
            text ([list(list(vector))]): list of lists of vector tokens
            labels ([list(int)]):list of labels
            meta_features (list(list(int))): list of metafetures, optional.

        Returns:
            [dataframe]: [vectorize dataset]
        """
        X = []
        y = []
        m = []
        for i in range(len(text)):
            sentence = text[i]
            embbedings = np.zeros(self.number_of_words * self.vector_length)
            for j in range(len(sentence)):
                c = j % self.number_of_words
                if (c == 0 and j > 0):
                    X.append(embbedings)
                    y.append(labels[i])
                    if (meta_features is not None):
                        m.append(meta_features[i])
                    embbedings = np.zeros(
                        self.number_of_words * self.vector_length)
                token = sentence[j]
                embbedings[c * self.vector_length:(c + 1) * self.vector_length] = self.pca.transform(
                    self.w2v.get_vector(token))

        if meta_features is not None:
            return np.array(X), np.array(y), np.array(m)
        else:
            return np.array(X), np.array(y)
    
    def to_string(self):
        return "Concat_{}".format(self.w2v.to_string())

""" This class implements the Vectorize class using a mean of vectors for each sample"""
class MeanW2V(Vectorizer):
    def __init__(self, core_w2v, vector_length):
        """Initiates the class

        Args:
            core_w2v (W2VCore): glove, reduced or gensim
            vector_length (integer): size of the reduced wanted vecor length
        """        
        super().__init__(vector_length)
        self.w2v = core_w2v

    def fit(self, text):
        """This function fits the PCA

        Args:
            text ([list(list(vector))]): list of lists of vector tokens
        """
        vectors = []
        for sentence in text:
            vectors.append(
                np.mean(self.w2v.text_to_vectors([sentence]), axis=0))

        vectors = np.array(vectors)
        self.pca.fit(vectors)

    def transform(self, text, labels,meta_features):
        """This function transform the dataset into vectorize manner with 
            mean of reduced size vectors

        Args:
            text ([list(list(vector))]): list of lists of vector tokens
            labels ([list(int)]):list of labels
            meta_features (list(list(int))): list of metafetures, optional.

        Returns:
            [dataframe]: [vectorize dataset]
        """        
        X = []
        y = []
        m = []
        for i in range(len(text)):
            sentence = text[i]
            vectors = []
            for token in sentence:
                vectors.append(self.w2v.get_vector(token))
            vectors = np.mean(vectors, axis=0)
            X.append(self.pca.transform(vectors).reshape(-1))
            y.append(labels[i])
            if (meta_features is not None):
                m.append(meta_features[i])
        
        if meta_features is not None:
            return np.array(X), np.array(y), np.array(m)
        else:
            return np.array(X), np.array(y)

    def to_string(self):
        return "Mean_{}".format(self.w2v.to_string())

""" This class implements the Vectorize class using TFIDF vectors for each sample"""
class TFIDF(Vectorizer):
    def __init__(self, vector_length):
        """Initiates the class

        Args:
            vector_length (integer): size of the reduced wanted vecor length
        """           
        super().__init__(vector_length)
        self.tfidf_vectorizer = TfidfVectorizer()

    def fit(self, text):
        """This function fits the PCA and sklearn TFIDF

        Args:
            text ([list(list(vector))]): list of lists of vector tokens
        """        
        X = np.asarray(self.tfidf_vectorizer.fit_transform(
            text.apply(lambda X: ' '.join(X))).todense())
        self.pca.fit(X)
        return None

    def transform(self, text, labels, meta_features):
        """This function transform the dataset into vectorize manner with 
            TFIDF reduced size vectors

        Args:
            text ([list(list(vector))]): list of lists of vector tokens
            labels ([list(int)]):list of labels
            meta_features (list(list(int))): list of metafetures, optional.

        Returns:
            [dataframe]: [vectorize dataset]
        """            
        X = self.pca.transform(np.array(self.tfidf_vectorizer.transform(
            text.apply(lambda X: ' '.join(X))).todense()))
        y = np.array(labels)
        
        return X, y

    def to_string(self):
        return "TFIDF"
