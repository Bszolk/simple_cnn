import numpy as np


class LossFunction:
    def __call__(self, y_hat: np.ndarray, y: np.ndarray, derivative: bool = False) -> np.ndarray:
        """
        Calculates the loss

        :param y_hat: model prediction
        :param y: correct label
        :param derivative: when True, calculates the derivative of the loss w.r.t. the y_hat
        :return: returns a tensor of losses for each sample in the batch
        """
        pass


class MeanSquaredError(LossFunction):
    def __call__(self, y_hat: np.ndarray, y: np.ndarray, derivative: bool = False) -> np.ndarray:
        if derivative:
            return 2 * (y_hat - y)
        return np.power((y_hat - y), 2)


class CrossEntropy(LossFunction):
    def __call__(self, y_hat: np.ndarray, y: np.ndarray, derivative: bool = False) -> np.ndarray:
        if derivative:
            return y_hat - y
        return -np.log(y_hat) * y
