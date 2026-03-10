import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from mnist_classifier.classifier_interface import MnistClassifierInterface
from tensorflow.keras.callbacks import EarlyStopping
class FeedForwardNNModel(MnistClassifierInterface):
    """
    The implemented model is a classic Feed-Forward Neural Network
    designed for efficient digit classification.  It begins with a Flatten layer that transforms
    the 2D input images (28x28 pixels) into 1D feature vectors. These vectors are processed by a
    fully connected hidden layer comprising 128 neurons with ReLU activation to capture non-linear relationships.
    The architecture concludes with a 10-unit Dense output layer using the Softmax activation function, which generates
    probability scores for each digit class, optimized using the Adam algorithm and sparse categorical cross-entropy loss.
    """
    def __init__(self):
        self.model = keras.Sequential([
            keras.Input(shape=(28, 28)),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, X_train, y_train, X_val=None, y_val=None):
        val_data = (X_val, y_val) if X_val is not None else None
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        history = self.model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            verbose=1,
            validation_data=val_data,
            callbacks=[early_stopping]
        )
        return history.history


    def predict(self, X_test):
        predictions = self.model.predict(X_test, verbose=0)
        return np.argmax(predictions, axis=1)