from layers import Layer, Linear, Convolutional, BatchNorm
from loss import LossFunction, MeanSquaredError, CrossEntropy
from optimizer import Adam, SGD
import numpy as np
import pickle


class Model:
    def __init__(self, layers: tuple[Layer, ...], loss_name: str = "MSE", batch_size: int = 32) -> None:
        self.layers: tuple[Layer, ...] = layers
        self.loss_f: LossFunction = ...
        self.set_loss_function(loss_name)
        self.batch_size: int = batch_size

        self.init_weights()

    @staticmethod
    def sequential(*layers: Layer) -> 'Model':
        """
        Creates a model from the sequence of layers

        :param layers: tuple of objects inheriting from Layer base class
        :return: model constructed from layer modules in the order they were passed
        """
        return Model(layers)

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def set_optimizer(self, optimizer_name: str, learning_rate: float) -> None:
        """
        Sets the optimizer for the whole model

        :param optimizer_name: name of the optimizer. Available: "Adam", "SGD"
        :param learning_rate: learning rate for the optimizer
        :return:
        """
        optimizer_name = optimizer_name.casefold()
        for layer in self.layers:
            if isinstance(layer, (Linear, Convolutional)):
                if optimizer_name == "Adam".casefold():
                    layer.set_optimizer(Adam(learning_rate))
                else:
                    layer.set_optimizer(SGD(learning_rate))

    def set_loss_function(self, loss_function: str) -> None:
        """
        Sets the loss function used by the model

        :param loss_function: name of the loss function. Available: "MSE", "CrossEntropy"
        :return:
        """
        loss_function = loss_function.casefold()
        if loss_function == "MSE".casefold():
            self.loss_f = MeanSquaredError()
        elif loss_function == "CrossEntropy".casefold():
            self.loss_f = CrossEntropy()
        else:
            raise ValueError("Incorrect loss function selected.")

    def init_weights(self) -> None:
        size = self.layers[0].size
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], (Linear, Convolutional)):
                self.layers[i].init_weights(size)
            if self.layers[i].size is not None:
                size = self.layers[i].size

    def prop_forward(self, x: np.ndarray) -> np.ndarray:
        y_hat = x
        for layer in self.layers:
            y_hat = layer.forward(y_hat)
        return y_hat

    def inference(self, x: np.ndarray) -> np.ndarray:
        y_hat = x
        for layer in self.layers:
            y_hat = layer.inference(y_hat) if isinstance(layer, BatchNorm) else layer.forward(y_hat)
        return y_hat

    def form_batches(self, n: int, drop_last: bool = False) -> list[np.ndarray]:
        """
        Creates a list of indexes that can be sampled from the dataset to form fixed length batches

        :param n: size of the dataset
        :param drop_last: whether to drop the last batch which can be smaller than the rest of batches
        :return: list of tensors containing indexes of the samples in given batch
        """
        indexes = np.arange(n)
        np.random.shuffle(indexes)
        batches = []
        i = 0
        limit = n if drop_last else n + self.batch_size
        while i + self.batch_size < limit:
            batches.append(indexes[i:i+128])
            i += self.batch_size
        return batches

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray,
              epochs: int) -> None:
        """
        Trains the model for given amount of epochs

        :param x_train: training input
        :param y_train: training labels (one-hot encoded)
        :param x_valid: validation input
        :param y_valid: validation labels (one-hot encoded)
        :param epochs: number of iterations over the whole training dataset
        :return:
        """
        for i in range(epochs):
            print(f"epoch {i+1}:")
            train_loss = self.fit(x_train, y_train)
            print(f"epoch {i+1} train loss: {train_loss:.4f}")
            valid_loss, accuracy = self.valid(x_valid, y_valid)
            print(f"epoch {i+1} valid loss: {valid_loss:.4f}")
            print(f"epoch {i+1} valid accuracy: {accuracy:.4f}")

    def fit(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Trains the model for one epoch

        :param x: training input
        :param y: training labels (one-hot encoded)
        :return: mean loss for the whole epoch
        """
        batches = self.form_batches(x.shape[0], drop_last=True)
        epoch_loss_sum = 0
        for batch_num, batch in enumerate(batches):
            print(f"batch: {batch_num}/{len(batches)}")
            x_batch = x[batch]
            y_batch = y[batch]

            y_hat = self.prop_forward(x_batch)
            loss = np.sum(self.loss_f(y_hat, y_batch)) / self.batch_size
            epoch_loss_sum += loss
            print(f"loss: {loss:.4f}")
            dloss = self.loss_f(y_hat, y_batch, derivative=True)

            for layer in reversed(self.layers):
                dloss = layer.backward(dloss)

            for j in range(1, len(self.layers)):
                gradient = self.layers[j].gradient(self.layers[j].dloss, self.layers[j - 1].out, self.batch_size)
                self.layers[j].optimize(gradient)
        return epoch_loss_sum / len(batches)

    def valid(self, x: np.ndarray, y: np.ndarray) -> (float, float):
        """
        Validate the model on the given data. Model compares it's predictions against the correct labels

        :param x: validation input
        :param y: validation labels (one-hot encoded)
        :return: tuple containing the mean validation loss and accuracy of the model
        """
        batches = self.form_batches(x.shape[0], drop_last=False)
        valid_loss_sum = 0
        correct = 0
        for batch_num, batch in enumerate(batches):
            x_batch = x[batch]
            y_batch = y[batch]

            y_hat = self.inference(x_batch)
            valid_loss_sum += np.sum(self.loss_f(y_hat, y_batch))
            predictions = np.argmax(y_hat, axis=1) == np.argmax(y_batch, axis=1)
            correct += np.sum(predictions)

        valid_loss = valid_loss_sum / x.shape[0]
        accuracy = correct / x.shape[0]
        return valid_loss, accuracy

    def save(self, file_name: str) -> None:
        pickle.dump(self, open(file_name, 'wb'))

    @staticmethod
    def load(file_name: str) -> 'Model':
        with open(file_name, 'rb') as f:
            return pickle.load(f, encoding='bytes')
