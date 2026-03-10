from sklearn.ensemble import RandomForestClassifier
from mnist_classifier.classifier_interface import MnistClassifierInterface

class RandomForestModel(MnistClassifierInterface):
    """
    Random Forest classifier for MNIST digit recognition,
    utilizing an ensemble of 300 decision trees to ensure accuracy and prevent overfitting.
    It includes an automated pipeline to flatten 28×28 images into 1D vectors and leverages
    multi-core parallel processing (n_jobs=-1) for rapid training.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=300, n_jobs=-1)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        flat_X = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(flat_X, y_train)

    def predict(self, X_test):
        flat_X = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(flat_X)
