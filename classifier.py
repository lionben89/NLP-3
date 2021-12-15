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
    def __init__(self, shape, n_epochs=50, criterion=nn.CrossEntropyLoss()):  # X_train_s.shape[1]
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(shape, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),

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

        train = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=False)

        for i in range(self.n_epochs):
            i += 1
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                y_pred = self.model.forward(inputs)  ## no batchs ?
                loss = self.criterion(y_pred, labels)
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

