from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    Abstract Base Class that enforces the contract for all specific model implementations.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model.
        X_train: Numpy array of shape (N, 28, 28) normalized to [0,1] preferably.
        y_train: Numpy array of labels (0-9).
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Predicts classes.
        X_test: Numpy array of shape (N, 28, 28).
        Returns: Numpy array of predicted integers (0-9).
        """
        pass