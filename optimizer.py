import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def optimize(self, weights, gradient):
        pass


class Default(Optimizer):
    def __init__(self, learning_rate=0.001):
        super().__init__(learning_rate)

    def optimize(self, weights, gradient):
        return weights - self.learning_rate * gradient


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = 0
        self.v = 0
        self.t = 0

    def optimize(self, weights, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient * gradient)
        m_hat = self.m / (1 - np.power(self.beta1, self.t))
        v_hat = self.v / (1 - np.power(self.beta2, self.t))

        return weights - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))
