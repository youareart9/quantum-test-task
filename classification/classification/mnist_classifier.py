from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from typing import Union

from utils import center_crop


class DigitClassificationInterface(ABC):
    """
    Interface for digit classification models.
    """
    @abstractmethod
    def predict(self, input_data: Union[np.ndarray, torch.Tensor]) -> int:
        """
        Predict the digit class based on the input data.

        :param input_data: Input data to the model.
            For CNN, it's a 28x28x1 tensor.
            For RandomForest, it's a 1D numpy array of length 784.
            For Random model, it's a 10x10 numpy array.
        :return: The predicted digit (integer).
        """
        pass

    @abstractmethod
    def train_model(self, X_train: Union[np.ndarray, torch.Tensor], y_train: np.ndarray):
        """
        Train the model given training data and labels.

        :param X_train: Training data (numpy array or PyTorch tensor).
        :param y_train: Training labels (numpy array).
        """
        pass


class CNNModel(DigitClassificationInterface):
    """
    Convolutional Neural Network (CNN) model for digit classification.
    """
    def __init__(self):
        """
        Initialize the CNN model with a predefined architecture.

        The architecture consists of:
        - 2 convolutional layers with ReLU activations and max pooling
        - A flatten layer
        - 2 fully connected (linear) layers

        The model is designed for classifying MNIST digits.
        """
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def predict(self, input_data: torch.Tensor) -> int:
        """
        Predict the digit from the input tensor using the CNN model.

        :param input_data: Input tensor of shape [28, 28, 1].
                           The tensor is assumed to be a single image with the shape
                           [height, width, channels].
        :return: Predicted digit (integer).
        """
        # Ensure the model is in evaluation mode
        self.model.eval()

        input_data = input_data.unsqueeze(0)  # Add batch dimension: [1, 28, 28, 1]
        input_data = input_data.permute(0, 3, 1, 2)  # Rearrange dimensions to [batch_size, channels, height, width]
        with torch.no_grad():
                input_data = input_data.float()  # Ensure input is float tensor
                output = self.model(input_data)
                _, predicted = torch.max(output, 1)
                return predicted.item()

    def train_model(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """
        Placeholder for training the model. Raises NotImplementedError since training is not
        implemented within this class.
        """
        raise NotImplementedError("Training is not implemented in CNNModel.")


class RandomForestModel(DigitClassificationInterface):
    """
    Random Forest model for digit classification.
    """
    def __init__(self):
        """
        Initialize the RandomForestModel with a RandomForestClassifier.

        The classifier is initialized with 100 estimators. This is a placeholder initialization;
        the model needs to be trained before making predictions.
        """
        self.model = RandomForestClassifier(n_estimators=100)  # Placeholder, actual model should be trained

    def predict(self, input_data: np.ndarray) -> int:
        """
        Predict the digit from the input numpy array using the RandomForest model.

        :param input_data: Input numpy array of shape [784,].
        :return: Predicted digit (integer).
        """
        return self.model.predict(input_data.reshape(1, -1))[0]

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Placeholder for training the model. Raises NotImplementedError since training is not
        implemented within this class.
        """
        raise NotImplementedError("Training is not implemented in RandomForestModel.")


class RandomModel(DigitClassificationInterface):
    """
    Random model for digit classification.
    """
    def __init__(self):
        """
        Initialize the RandomModel.

        This model does not require any specific initialization or training.
        It is a placeholder model that provides random predictions.
        """
        pass

    def predict(self, input_data: np.ndarray) -> int:
        """
        Predict a random digit from the input numpy array using the Random model.

        :param input_data: Input numpy array of shape (10, 10) or any shape. The input is not used
                           in the prediction process.
        :return: A randomly generated digit between 0 and 9 (inclusive).
        """
        return np.random.randint(0, 10)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Placeholder for training the model. Raises NotImplementedError since training is not
        implemented within this class.
        """
        raise NotImplementedError("Training is not implemented in RandomModel.")


class DigitClassifier:
    """
    Digit classifier that uses different algorithms to predict digits.
    """
    def __init__(self, algorithm: str):
        """
        Initialize the DigitClassifier with the specified algorithm.

        :param algorithm: The name of the algorithm to use ('cnn', 'rf', 'rand').
        :raises ValueError: If the provided algorithm is not supported.
        """
        self.algorithm = algorithm
        # Initialize the model based on the selected algorithm
        if algorithm == 'cnn':
            self.model = CNNModel()
        elif algorithm == 'rf':
            self.model = RandomForestModel()
        elif algorithm == 'rand':
            self.model = RandomModel()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train_model(self):
        """
        Placeholder for training the model. Raises NotImplementedError since training is not
        implemented within this class.
        """
        raise NotImplementedError("Training is not implemented in DigitClassifier.")

    def predict(self, input_data: np.ndarray) -> int:
        """
        Predict the digit using the selected algorithm.

        :param input_data: Input image as a numpy array with shape [28, 28, 1].
        :return: Predicted digit (integer).
        """
        if self.algorithm == 'cnn':
            input_data = torch.tensor(input_data, dtype=torch.float32)
            return self.model.predict(input_data)
        elif self.algorithm == 'rf':
            input_data = input_data[:, :, 0].flatten()
            return self.model.predict(input_data)
        elif self.algorithm == 'rand':
            input_data = input_data[:, :, 0]
            cropped_image = center_crop(input_data, (10, 10))  # Adjust crop size as needed
            return self.model.predict(cropped_image)
