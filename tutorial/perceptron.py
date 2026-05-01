from abc import ABC, abstractmethod


class Perceptron(ABC):
    def __init__(self, learning_rate=0.1, epochs=20,epsilon=0.02):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []
        self.epsilon = epsilon

    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        ...
