from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from torchnlp.word_to_vector import GloVe
from sklearn.decomposition import PCA
import numpy as np

class Vectorizer(ABC):
    def __init__(self,vector_length=200):
        super().__init__()
        self.vector_length = vector_length
        self.pca = PCA(n_components=vector_length)
        
    
    @abstractmethod
    def fit(self, text):
        pass
    
    @abstractmethod
    def transform(self,text,labels):
        pass
    
    def fit_transform(self,text,labels):
        self.fit(text)
        return self.transform(text,labels)


## not working yet    
class Glove(Vectorizer):
    def __init__(self,vector_length,number_of_words=10, filename):
        super().__init__(vector_length)
        self.w2v_glove = GloVe(name=filename)
        self.number_of_words = number_of_words
        
    def fit(self,text):
        self.pca.fit(self.w2v_glove)
        
    def save():
        return None

    def transform(self, text, labels):
        X = []
        y = []
        for i in range(len(text)):
            sentence = text[i]
            embbedings = np.zeros(self.number_of_words * self.vector_length)
            for j in range(len(sentence)):
                if (j % self.number_of_words == 0 and j > 0):
                    X.append(embbedings)
                    y.append(labels[i])
                    embbedings = np.zeros(self.number_of_words * self.vector_length)
                token = sentence[j]
                embbedings[j*self.vector_length:(j+1)*self.vector_length] = self.pca.transform(self.w2v_glove[token])
        return X,y
    

        

class TFIDF(Vectorizer):
    def __init__(self,vector_length):
        super().__init__(vector_length)
        self.tfidf_vectorizer = TfidfVectorizer()
        
    
    def fit(self,text):
        X = np.asarray(self.tfidf_vectorizer.fit_transform(text.apply(lambda X: ' '.join(X))).todense())
        self.pca.fit(X)
        return None
    
    def transform(self, text, labels):
        X = self.pca.transform(np.asarray(self.tfidf_vectorizer.transform(text.apply(lambda X: ' '.join(X))).todense()))
        y = labels
        return X, y

