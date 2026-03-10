from models.nn_model import FeedForwardNNModel
from models.cnn_model import CNNModel
from models.rf_model import RandomForestModel

class MnistClassifier:
    """
    Wrapper class that hides the specific algorithm implementation.
    """
    def __init__(self, algorithm: str):
        self.algorithm = algorithm.lower()
        if self.algorithm == 'rf':
            self._model = RandomForestModel()
        elif self.algorithm == 'nn':
            self._model = FeedForwardNNModel()
        elif self.algorithm == 'cnn':
            self._model = CNNModel()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Training {self.algorithm.upper()} model...")
        history = self._model.train(X_train, y_train, X_val, y_val)
        print("Training complete")
        return history

    def predict(self, X_test):
        return self._model.predict(X_test)