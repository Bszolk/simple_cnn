from optimizer import Optimizer
import numpy as np
from typing import Callable, Any


class Layer:
    """
    Base abstract class for all layers
    """
    def __init__(self) -> None:
        self.size = None
        self.out = None
        self.dloss = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates forward pass of the layer

        :param x: output tensor of the previous layer
        :return: returns the result of the forward pass
        """
        pass

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        """
        Calculates derivative of the loss w.r.t. the layer's input

        :param dloss: derivative of the loss w.r.t. the layer's output
        :return: returns the derivative of the loss w.r.t. the layer's input
        """
        pass

    def gradient(self, dloss: np.ndarray, out: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Calculates the gradient w.r.t. the layer's weights

        :param dloss: derivative of the loss w.r.t. the layer's output
        :param out: output of the previous layer
        :param batch_size: currently used batch size
        :return: returns the gradient
        """
        pass

    def optimize(self, gradient: np.ndarray) -> None:
        """
        Uses optimizer assigned to the layer to update the weights

        :param gradient: calculated gradient w.r.t. the layer's weights
        :return:
        """
        pass


class Linear(Layer):
    """
    Linear (densely connected) layer. XW.T + b
    """
    def __init__(self, size: int) -> None:
        """
        :param size: number of weights in the layer
        """
        super().__init__()
        self.size: int = size
        self.optimizer: Optimizer = Optimizer()

        self.weights: np.ndarray = np.array([])
        self.bias: np.ndarray = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.dot(x, self.weights) + self.bias
        return self.out

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        self.dloss = dloss
        return self.dloss @ self.weights.transpose()

    def optimize(self, gradient: np.ndarray) -> None:
        self.weights = self.optimizer.optimize(self.weights, gradient)
        avg = np.average(self.dloss, keepdims=True, axis=0)
        self.bias += -self.optimizer.learning_rate * avg

    def gradient(self, dloss: np.ndarray, out: np.ndarray, batch_size: int) -> np.ndarray:
        return (out.transpose() @ dloss) / batch_size

    def set_optimizer(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def init_weights(self, from_layer_size: int) -> None:
        """
        He / Kaiming normal initialization of the layer's weights

        :param from_layer_size: size of the previous layer
        :return:
        """
        n = from_layer_size
        if isinstance(n, tuple):
            n = np.prod(from_layer_size)
        limit = np.sqrt(2 / n)
        self.weights = np.random.normal(0, limit, size=(n, self.size))
        self.bias = np.zeros((1, self.size), dtype=np.float64)


class Input(Layer):
    """
    Always the first layer in the model. Specifies the shape of the data inputted into the model
    """
    def __init__(self, size: Any) -> None:
        super().__init__()
        self.size: Any = size

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = x  # necessary for backpropagation to work
        return x


class Convolutional(Layer):
    """
    Convolutional layer "slides" a kernel of weights over an input image and tries to extract meaningful features
    """
    def __init__(self, size: tuple, kernel_size: tuple, padding: str = "same",
                 stride: int = 1, kernel_n: int = 1) -> None:
        """
        :param size: size of input image. 3d tuple(channels, height, width)
        :param kernel_size: 3d tuple(channels, height, width). height and width should be odd
        :param padding: padding type.
        "valid" - no padding, image size decreases by (kernel_height/width - 1),
        "same" - padding added, image size stays the same
        :param stride: by how much the kernel is slided each step. 1 is most commonly used
        :param kernel_n: number of kernels used (number of channels in the output image)
        """
        super().__init__()

        self.size: tuple = size
        if len(size) != 3:
            raise ValueError("size parameter should be a tuple(k, m, n) where k indicates the number of channels and "
                             "m, n indicate the shape of the image")
        self.channels: int = size[0]
        self.img_shape: tuple = size[1:]
        self.pad_img_shape: tuple = size[1:]

        self.ker_channels: int = kernel_size[0]
        self.ker_shape: tuple = kernel_size[1:]
        self.ker_n: int = kernel_n

        self.padding: str = padding
        self.pad_horizontal: int = 0
        self.pad_vertical: int = 0

        self.stride: int = stride

        self.kernels: np.ndarray = np.ndarray([])
        self.biases: np.ndarray = np.ndarray([])

        self.optimizer: Optimizer = Optimizer()

        if self.padding == "same":
            if self.ker_shape[0] % 2 == 0 or self.ker_shape[1] % 2 == 0:
                raise ValueError("'same' padding option cannot be used with even kernel shape")
            self.pad_horizontal = (self.ker_shape[1] - 1) // 2
            self.pad_vertical = (self.ker_shape[0] - 1) // 2
            self.pad_img_shape = (self.pad_img_shape[0] + self.pad_vertical * 2,
                                  self.pad_img_shape[1] + self.pad_horizontal * 2)

        if (self.img_shape[1] - self.ker_shape[1]) % stride != 0 or (self.img_shape[0] - self.ker_shape[0]) % stride != 0:
            raise ValueError("selected stride value cannot be used with given image and kernel shape")

    def set_optimizer(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def init_weights(self, from_layer_size: int) -> None:
        """
        He / Kaiming normal initialization of the layer's weights

        :param from_layer_size: size of the previous layer
        :return:
        """
        n = from_layer_size
        if isinstance(n, tuple):
            n = np.prod(from_layer_size)
        limit = np.sqrt(2 / n)
        self.kernels = np.random.normal(0, limit,
                                        size=(self.ker_n, self.ker_channels, self.ker_shape[0], self.ker_shape[1]))
        self.biases = np.zeros(self.ker_n, dtype=np.float64).reshape((self.ker_n, 1))

    def convolve(self, x: np.ndarray, kernels: np.ndarray, stride: int, flip: bool = True) -> np.ndarray:
        """
        :param x: input image of shape [batch_size, 1, img_channels, img_height, img_width]
        :param kernels: kernels of shape [1, kernel_n, ker_channels, ker_height, ker_width]
        :param stride: by how much the kernel is slided each step
        :param flip: whether to flip the kernel matrix by 180
        :return: activation map of shape [batch_size, kernel_n, img_channels, new_img_height, new_img_width]
        """
        batch = max(x.shape[0], kernels.shape[0])
        channels = max(x.shape[2], kernels.shape[2])
        x_channels = x.shape[2]
        x_shape = x.shape[-2:]
        n_kernels = kernels.shape[1]

        # ker_channels == x_channels
        ker_channels = kernels.shape[2]
        assert x_channels == ker_channels or x_channels == 1 or ker_channels == 1
        ker_shape = kernels.shape[-2:]

        # row_index[:, i] indicates row indexes of the input to be filtered with a kernel on i-th iteration
        i = np.repeat(np.arange(x_shape[0] - ker_shape[0] + 1, step=stride),
                      (x_shape[1] - ker_shape[1]) // stride + 1)
        row_index = np.repeat(i.reshape((1, -1)) + np.arange(ker_shape[0]).reshape((-1, 1)), ker_shape[1],
                              axis=0)

        # col_index[:, i] indicates column indexes of the input to be filtered with a kernel on i-th iteration
        j = np.tile(np.arange(x_shape[1] - ker_shape[1] + 1, step=stride),
                    (x_shape[0] - ker_shape[0]) // stride + 1)
        col_index = np.tile(j.reshape((1, -1)) + np.arange(ker_shape[1]).reshape((-1, 1)), (ker_shape[0], 1))

        if flip:
            kernels = np.rot90(kernels, 2, (3, 4))

        # (1, self.ker_n, self.channels, 1, ker_w * ker_h) @ (batch, 1, self.channels, ker_w * ker_h, img_w * img_h)
        activation_map = kernels.reshape((kernels.shape[0], n_kernels, kernels.shape[2], 1, -1)) @ x[:, :, :, row_index, col_index]

        new_height = (x_shape[0] - ker_shape[0]) // stride + 1
        new_width = (x_shape[1] - ker_shape[1]) // stride + 1
        activation_map = activation_map.reshape((batch, n_kernels, channels, new_height, new_width))
        return activation_map

    def forward(self, x: np.ndarray) -> np.ndarray:
        img = np.pad(x, ((0, 0), (0, 0),
                         (self.pad_vertical, self.pad_vertical), (self.pad_horizontal, self.pad_horizontal)))

        img = img[:, np.newaxis, :, :, :]
        kernels = self.kernels[np.newaxis, :, :, :, :]
        activation_map = self.convolve(img, kernels, self.stride, flip=True)
        # summing all channels together
        activation_map = np.sum(activation_map, axis=2)

        # adding bias
        activation_map_w_bias = activation_map + self.biases[np.newaxis, :, :, np.newaxis]
        self.out = activation_map_w_bias
        return self.out

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        self.dloss = dloss

        # applying padding to return to the same shape from before convolution
        pad_h = (self.ker_shape[0] - 1) // 2
        pad_w = (self.ker_shape[1] - 1) // 2
        if self.padding == "valid":
            pad_h += (self.ker_shape[0] - 1) // 2
            pad_w += (self.ker_shape[1] - 1) // 2
        dloss = np.pad(self.dloss, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))

        dloss = dloss[:, :, np.newaxis, :, :]
        kernels = self.kernels[np.newaxis, :, :, :, :]

        m = self.convolve(dloss, kernels, self.stride, flip=True)
        m = np.sum(m, axis=1)
        return m

    def gradient(self, dloss: np.ndarray, out: np.ndarray, batch_size: int) -> np.ndarray:
        out = out[:, np.newaxis, :, :, :]
        dloss = dloss[:, :, np.newaxis, :, :]
        kernels_gradient = self.convolve(out, dloss, self.stride, flip=True)
        kernels_gradient = np.sum(kernels_gradient, axis=0)
        kernels_gradient = kernels_gradient / batch_size
        return kernels_gradient

    def optimize(self, gradient: np.ndarray) -> None:
        self.optimizer.optimize(self.kernels, gradient)

        avg = np.average(np.average(np.average(self.dloss, axis=2), axis=2), axis=0, keepdims=True)
        self.biases += -self.optimizer.learning_rate * avg.transpose()


class Pooling(Layer):
    """
    Shrinks the image size while keeping the most important information
    """
    def __init__(self, size: tuple, filter_size: tuple = (2, 2), pooling_type: str = "max") -> None:
        """
        :param size: size of input image. 3d tuple(channels, height, width)
        :param filter_size: size of the filter. 2d tuple(height, width). default is (2,2)
        :param pooling_type: "max" - for each filter step only the max value is kept
        """
        super(Pooling, self).__init__()
        self.size: tuple = size
        self.img_shape: tuple = size[-2:]
        self.pooling_type: str = pooling_type
        if self.pooling_type == "max":
            self.pooling_f: Callable[[np.ndarray], Any] = np.max
        else:
            raise ValueError(f"unknown pooling type")

        self.filter_size: tuple = filter_size
        self.v_stride: int = filter_size[0]
        self.h_stride: int = filter_size[1]

        # row_index[:, i] indicates row indexes of the input to be filtered with a filter on i-th iteration
        i = np.repeat(np.arange(self.img_shape[0] - self.filter_size[0] + 1, step=self.v_stride),
                      (self.img_shape[1] - self.filter_size[1]) // self.h_stride + 1)
        self.row_index = np.repeat(i.reshape((1, -1)) + np.arange(self.filter_size[0]).reshape((-1, 1)),
                                   self.filter_size[1], axis=0)

        # col_index[:, i] indicates column indexes of the input to be filtered with a filter on i-th iteration
        j = np.tile(np.arange(self.img_shape[1] - self.filter_size[1] + 1, step=self.h_stride),
                    (self.img_shape[0] - self.filter_size[0]) // self.v_stride + 1)
        self.col_index = np.tile(j.reshape((1, -1)) + np.arange(self.filter_size[1]).reshape((-1, 1)),
                                 (self.filter_size[0], 1))

        self.mask: np.ndarray = np.array([])

        if len(size) != 3:
            raise ValueError("size parameter should be a tuple (k, m, n) where k indicates the number of channels and "
                             "m, n indicate the shape of the image")

        if (self.img_shape[1] - self.filter_size[1]) % self.h_stride != 0 or \
                (self.img_shape[0] - self.filter_size[0]) % self.v_stride != 0:
            raise ValueError("selected filter cannot be used with this image size")

    def forward(self, x: np.ndarray) -> np.ndarray:
        pool = self.pooling_f(x[:, :, self.row_index, self.col_index], axis=2)

        if self.pooling_type == "max":
            x[:, :, self.row_index, self.col_index] = (x[:, :, self.row_index, self.col_index] == pool[:, :, np.newaxis, :]) * 1
            self.mask = x

        pool = pool.reshape((x.shape[0], x.shape[1],
                             (self.img_shape[0] - self.filter_size[0]) // self.v_stride + 1,
                             (self.img_shape[1] - self.filter_size[1]) // self.h_stride + 1))

        self.out = pool
        return pool

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        dloss = dloss.reshape((dloss.shape[0], dloss.shape[1], 1, -1))
        z = np.zeros_like(self.mask)
        z[:, :, self.row_index, self.col_index] = self.mask[:, :, self.row_index, self.col_index] * dloss
        return z


class Flatten(Layer):
    """
    Flattens the input tensor while keeping the same batch size. for example 4d tuple(B, A, B, C) -> 2d tuple(B, A*B*C)
    """
    def __init__(self, size: tuple) -> None:
        """
        :param size: size of the tensor to be flattened (without batch dimension)
        """
        super(Flatten, self).__init__()
        self.size: tuple = size
        self.batch_size: int = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.batch_size = x.shape[0]
        x = x.reshape((x.shape[0], -1))
        self.out = x
        return x

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        return dloss.reshape((self.batch_size, self.size[0], self.size[1], self.size[2]))


class ReLU(Layer):
    """
    ReLU activation function
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.maximum(0, x)
        return self.out

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        return (self.out > 0) * dloss


class SoftMax(Layer):
    """
    Softmax activation function
    """
    def __init__(self) -> None:
        super().__init__()

    def softmax(self, x: np.ndarray) -> np.ndarray:
        x_scaled = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x_scaled)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = self.softmax(x)
        return self.out

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        sm = self.softmax(self.out)
        jac = np.eye(sm.shape[1], sm.shape[1])[np.newaxis, :, :]
        jac = jac - sm[:, np.newaxis, :]
        jac = jac * sm[:, :, np.newaxis]
        return np.sum(jac * dloss[:, :, np.newaxis], axis=1)


class BatchNorm(Layer):
    """
    Normalizes each feature in the input batch
    """
    def __init__(self) -> None:
        super().__init__()

        self.x: np.ndarray = np.array([])
        self.m: int = 0
        self.n: int = 0
        self.initial_shape: tuple = tuple()
        self.mean: np.ndarray = np.array([])
        self.std: np.ndarray = np.array([])

        # beta, gamma - learnable parameters
        self.beta: np.ndarray = np.array([[0]])
        self.gamma: np.ndarray = np.array([[1]])
        self.normalized: np.ndarray = np.array([])
        self.epsilon: float = 1e-8

        self.mean_ma: Any = np.array([0])  # mean moving average
        self.var_ma: Any = np.array([0])  # variance moving average
        self.momentum: float = 0.9

        self.learning_rate: float = 0.001

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.initial_shape = x.shape
        self.m = x.shape[0]
        self.n = np.prod(x.shape[1:])
        if len(x.shape) > 2:
            self.x = x.reshape((self.m, self.n))

        self.mean = np.sum(self.x, axis=0, keepdims=True) / self.m
        self.std = np.sqrt(np.sum(np.power(self.x - self.mean, 2), axis=0, keepdims=True) / self.m)

        self.normalized = (self.x - self.mean) / np.sqrt((np.power(self.std, 2) + self.epsilon))

        self.out = self.gamma * self.normalized + self.beta
        self.out = self.out.reshape(self.initial_shape)

        self.mean_ma = self.momentum * self.mean_ma + (1 - self.momentum) * self.mean
        self.var_ma = self.momentum * self.var_ma + (1 - self.momentum) * np.power(self.std, 2)

        return self.out

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        if len(dloss.shape) > 2:
            dloss = dloss.reshape((self.m, self.n))

        dnormalized = dloss * self.gamma
        dstd = np.sum(dnormalized.transpose() @ ((self.x - self.mean) * (-1 / 2) * np.power((np.power(self.std, 2) + self.epsilon), (-3 / 2))), axis=0)
        dmean = dnormalized * (-1 / np.sqrt(np.power(self.std, 2) + self.epsilon))

        self.beta = self.beta - self.learning_rate * np.sum(dloss, axis=0)
        self.gamma = self.gamma - self.learning_rate * np.sum(dloss.transpose() @ self.normalized, axis=1)

        dx = dnormalized * (1 / np.sqrt(np.power(self.std, 2) + self.epsilon)) + dstd * (2 * (self.x - self.mean) / self.m) + dmean * (1 / self.m)
        dx = dx.reshape(self.initial_shape)
        return dx

    def inference(self, x: np.ndarray) -> np.ndarray:
        m = x.shape[0]
        n = np.prod(x.shape[1:])
        initial_shape = x.shape
        if len(x.shape) > 2:
            x = x.reshape((m, n))

        # pre-recorded mean and variance used during inference
        normalized = (x - self.mean_ma) / np.sqrt((self.var_ma + self.epsilon))
        transformed = self.gamma * normalized + self.beta
        transformed = transformed.reshape(initial_shape)
        return transformed
