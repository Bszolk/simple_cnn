import numpy as np


class Optimizer:
    """
    Used to update weights of the model
    """
    def __init__(self, learning_rate: float = 0.001) -> None:
        self.learning_rate: float = learning_rate

    def optimize(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Calculates and returns new layer's weights

        :param weights: weights of the model to be updated
        :param gradient: gradient of the loss w.r.t. the layer's weights
        :return: new weights, after the update
        """
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.001) -> None:
        super().__init__(learning_rate)

    def optimize(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return weights - self.learning_rate * gradient


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8) -> None:
        super().__init__(learning_rate)
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon

        self.m: np.ndarray = np.array([0])
        self.v: np.ndarray = np.array([0])
        self.t: int = 0

    def optimize(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient * gradient)
        m_hat = self.m / (1 - np.power(self.beta1, self.t))
        v_hat = self.v / (1 - np.power(self.beta2, self.t))

        return weights - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))
