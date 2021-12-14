from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import torch.nn as nn


class Classifier(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_test, Y_test):
        pass

    @abstractmethod
    def get_cls(self):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass


class LRClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.logreg = LogisticRegression()

    def get_cls(self):
        return self.logreg

    def train(self, X_train, y_train):
        self.logreg.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.logreg.predict(X_test)

    def evaluate(self, X_test, Y_test):
        pass


class SVMClassifier(Classifier):
    def __init__(self, kernel='liner', gamma='scale'):
        super().__init__()
        self.svm = SVC(kernel=kernel, gamma=gamma)

    def get_cls(self):
        return self.svm

    def train(self, X_train, y_train):
        self.svm.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.svm.predict(X_test)

    def evaluate(self, X_test, Y_test):
        pass


class BasicNN(Classifier):
    def __init__(self, input_shape, n_epochs=50, criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.n_epochs = n_epochs
        self.losses = []

    def get_cls(self):
        return self.model

    def train(self, X_train, y_train):
        X_train = torch.Tensor(X_train)
        y_train = torch.LongTensor(y_train)

        for i in range(self.n_epochs):
            y_pred = self.model.forward(X_train)  # no batchs ?
            loss = self.criterion(y_pred, y_train)
            self.losses.append(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return None

    def predict(self, X_test):
        tensor_pred = self.model(torch.Tensor(X_test))
        return tensor_pred.max(1).indices

    def evaluate(self, X_test, Y_test):
        pass


class LSTMNET(nn.Module):
    def __init__(self, input_shape ,n_layers, linear_dim, dropout=0.5):
        super(LSTMNET, self).__init__()
        self.n_layers = n_layers
        self.linear_dim = linear_dim
        
        self.lstm = nn.LSTM(input_shape, linear_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(linear_dim, 2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, y_tag):
        batch_size = 1
        x = x.long()
        if (y_tag is None):
            y_tag = torch.zeros(self.n_layers, batch_size, self.linear_dim)
        lstm_out, y_tag = self.lstm(x, y_tag)
        lstm_out = lstm_out.contiguous().view(-1, self.linear_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, y_tag  


class LSTMClassifier(Classifier):
    def __init__(self,  input_shape ,n_layers, linear_dim, dropout=0.5, n_epochs=50, criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.model = LSTMNET(input_shape ,n_layers, linear_dim, dropout)
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.n_epochs = n_epochs
        self.losses = []

    def get_cls(self):
        return self.model

    def train(self, X_train, y_train):
        X_train = torch.Tensor(X_train)
        y_train = torch.LongTensor(y_train)
        y_tag = None
        for i in range(self.n_epochs):
            y_pred, y_tag = self.model.forward(X_train,y_tag)  # no batchs ?
            loss = self.criterion(y_pred, y_train)
            self.losses.append(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return None

    def predict(self, X_test):
        tensor_pred = self.model(torch.Tensor(X_test))
        return tensor_pred.max(1).indices

    def evaluate(self, X_test, Y_test):
        pass