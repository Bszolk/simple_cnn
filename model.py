from layers import Layer, Linear, Convolutional, BatchNorm
from loss import mean_squared_error, categorical_cross_entropy
from optimizer import Optimizer, Adam, Default
from typing import Callable
import numpy as np
import pickle


class Model:

    def __init__(self, layers: tuple[Layer], loss_name: str = "mean_squared_error", batch_size: int = 32):
        self.layers: tuple[Layer] = layers
        self.loss_f: Callable = self.set_loss_function(loss_name)
        self.batch_size: int = batch_size

        self.init_weights()

    @staticmethod
    def create(*layers: Layer):
        return Model(layers)

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def set_optimizer(self, optimizer_name: str, learning_rate: float):
        for layer in self.layers:
            if isinstance(layer, Linear) or isinstance(layer, Convolutional):
                if optimizer_name == "adam":
                    layer.set_optimizer(Adam(learning_rate))
                else:
                    layer.set_optimizer(Default(learning_rate))

    def set_loss_function(self, loss_function):
        if loss_function == "mean_squared_error":
            self.loss_f = mean_squared_error
        elif loss_function == "categorical_cross_entropy":
            self.loss_f = categorical_cross_entropy
        else:
            raise ValueError("Incorrect loss function selected.")

    def init_weights(self):
        size = self.layers[0].size
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], Linear) or isinstance(self.layers[i], Convolutional):
                self.layers[i].init_weights(size)
            if self.layers[i].size is not None:
                size = self.layers[i].size

    def prop_forward(self, x):
        y_hat = x
        for layer in self.layers:
            y_hat = layer.forward(y_hat)
        return y_hat

    def inference(self, x):
        y_hat = x
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                y_hat = layer.inference(y_hat)
            else:
                y_hat = layer.forward(y_hat)
        return y_hat

    def fit(self, x, y, epochs):
        for i in range(epochs):
            print(f"epoch: {i}")
            batch_index = np.random.choice(np.arange(x.shape[0]), self.batch_size, replace=False)
            batch = x[batch_index]
            y_batch = y[batch_index]

            y_hat = self.prop_forward(batch)
            loss = np.sum(self.loss_f(y_hat, y_batch)) / self.batch_size
            print(f"loss: {loss}")
            dloss = self.loss_f(y_hat, y_batch, derivative=True)

            for layer in reversed(self.layers):
                # print(dloss.shape)
                dloss = layer.backward(dloss)

            for j in range(1, len(self.layers)):
                gradient = self.layers[j].gradient(self.layers[j].dloss, self.layers[j - 1].out, self.batch_size)
                self.layers[j].optimize(gradient)

    def validate(self, x, y):
        batch_size = 128
        indexes = np.arange(x.shape[0])
        np.random.shuffle(indexes)
        batch_start = 0
        batch_end = min(batch_size, x.shape[0])

        tp = 0
        while batch_start < batch_end:
            batch = x[indexes[batch_start:batch_end]]
            y_hat = self.inference(batch)
            y_hat_classified = np.argmax(y_hat, axis=1)
            y_classified = np.argmax(y[indexes[batch_start:batch_end]], axis=1)
            tp += np.sum(y_hat_classified == y_classified)

            batch_start += batch_size
            batch_end = min(batch_end + batch_size, x.shape[0])

        print(f"acc: {tp/y.shape[0]}")

    def predict(self, x):
        y_hat = self.inference(x)
        y_hat_classified = np.argmax(y_hat, axis=0)
        return y_hat_classified

    def save(self, file_name):
        pickle.dump(self, open(file_name, 'wb'))

    @staticmethod
    def load(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f, encoding='bytes')
