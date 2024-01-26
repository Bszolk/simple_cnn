from optimizer import Optimizer, Default, Adam
import numpy as np


class Layer:
    def __init__(self):
        self.size = None

        self.out = None
        self.dloss = None

    def forward(self, x):
        pass

    def backward(self, dloss):
        pass

    def gradient(self, dloss, out, batch_size):
        pass

    def optimize(self, gradient):
        pass


class Linear(Layer):
    def __init__(self, size: int, optimizer: Optimizer = Default()):
        super().__init__()
        self.size = size
        self.optimizer = optimizer

        self.weights = None
        self.bias = None

    def forward(self, x):
        self.out = np.dot(x, self.weights)
        return self.out

    def backward(self, dloss):
        self.dloss = dloss
        return self.dloss @ self.weights.transpose()

    def optimize(self, gradient):
        self.weights = self.optimizer.optimize(self.weights, gradient)
        avg = np.average(self.dloss, keepdims=True, axis=0)
        self.bias += -self.optimizer.learning_rate * avg

    def gradient(self, dloss, out, batch_size):
        return (out.transpose() @ dloss) / batch_size

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def init_weights(self, from_layer_size):
        np.random.seed(0)
        n = from_layer_size
        if isinstance(n, tuple):
            n = np.prod(from_layer_size)
        limit = np.sqrt(2 / n)
        self.weights = np.random.normal(0, limit, size=(n, self.size))
        self.bias = np.zeros((1, self.size), dtype=np.float64)


class Input(Layer):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        self.out = x
        return x

    def backward(self, dloss):
        return None

    def gradient(self, dloss, out, batch_size):
        return None

    def optimize(self, gradient):
        pass


class Convolutional(Layer):
    def __init__(self, size: tuple, kernel_size: tuple, padding: str = "same", stride: int = 1, kernel_n: int = 3,
                 optimizer: Optimizer = Adam(0.001)):
        super().__init__()

        self.size = size
        if len(size) != 3:
            raise ValueError("size parameter should be a tuple (k, m, n) where k indicates the number of channels and "
                             "m, n indicate the shape of the image")
        self.channels = size[0]
        self.img_shape = size[1:]
        self.pad_img_shape = size[1:]

        self.ker_channels = kernel_size[0]
        self.ker_shape = kernel_size[1:]
        self.ker_n = kernel_n

        self.padding = padding
        self.pad_horizontal = 0
        self.pad_vertical = 0

        self.stride = stride

        self.kernels = None
        self.biases = None

        self.optimizer = optimizer

        if self.padding == "same":
            if self.ker_shape[0] % 2 == 0 or self.ker_shape[1] % 2 == 0:
                raise ValueError("'same' padding option cannot be used with even kernel shape")
            self.pad_horizontal = (self.ker_shape[1] - 1) // 2
            self.pad_vertical = (self.ker_shape[0] - 1) // 2
            self.pad_img_shape = (self.pad_img_shape[0] + self.pad_vertical * 2,
                                  self.pad_img_shape[1] + self.pad_horizontal * 2)

        if (self.img_shape[1] - self.ker_shape[1]) % stride != 0 or \
                (self.img_shape[0] - self.ker_shape[0]) % stride != 0:
            raise ValueError("selected stride value cannot be used with given image and kernel shape")

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def init_weights(self, from_layer_size):
        n = from_layer_size
        if isinstance(n, tuple):
            n = np.prod(from_layer_size)
        limit = np.sqrt(2 / n)
        # np.random.seed(0)
        self.kernels = np.random.normal(0, limit,
                                        size=(self.ker_n, self.ker_channels, self.ker_shape[0], self.ker_shape[1]))
        self.biases = np.zeros(self.ker_n, dtype=np.float64).reshape((self.ker_n, 1))

    # x, kernels - 5D
    def cross_correlate(self, x, kernels, stride, convolve=False):
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

        if convolve:
            kernels = np.rot90(kernels, 2, (3, 4))

        # (1, self.ker_n, self.channels, 1, ker_w * ker_h) @ (batch, 1, self.channels, ker_w * ker_h, img_w * img_h)
        activation_map = kernels.reshape((kernels.shape[0], n_kernels, kernels.shape[2], 1, -1)) @ \
                         x[:, :, :, row_index, col_index]

        activation_map = activation_map.reshape(
            (batch,
             n_kernels,
             channels,
             (x_shape[0] - ker_shape[0]) // stride + 1,
             (x_shape[1] - ker_shape[1]) // stride + 1)
        )
        return activation_map

    def forward(self, x):
        img = np.pad(x, ((0, 0),
                         (0, 0),
                         (self.pad_vertical, self.pad_vertical),
                         (self.pad_horizontal, self.pad_horizontal)))

        activation_map = self.cross_correlate(img[:, np.newaxis, :, :, :],
                                              self.kernels[np.newaxis, :, :, :, :],
                                              self.stride,
                                              convolve=True)
        # summing all channels together
        activation_map = np.sum(activation_map, axis=2)

        # adding bias
        activation_map_w_bias = activation_map + self.biases[np.newaxis, :, :, np.newaxis]
        self.out = activation_map_w_bias
        return self.out

    def backward(self, dloss):
        self.dloss = dloss

        pad_h = (self.ker_shape[0] - 1) // 2
        pad_w = (self.ker_shape[1] - 1) // 2
        if self.padding == "valid":
            pad_h += (self.ker_shape[0] - 1) // 2
            pad_w += (self.ker_shape[1] - 1) // 2
        dloss = np.pad(self.dloss, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))

        dloss = dloss[:, :, np.newaxis, :, :]
        kernels = self.kernels[np.newaxis, :, :, :, :]

        m = self.cross_correlate(dloss, kernels, self.stride, convolve=True)
        m = np.sum(m, axis=1)
        return m

    def gradient(self, dloss, out, batch_size):
        kernels_gradient = self.cross_correlate(out[:, np.newaxis, :, :, :],
                                                dloss[:, :, np.newaxis, :, :],
                                                self.stride,
                                                convolve=True)
        kernels_gradient = np.sum(kernels_gradient, axis=0)
        kernels_gradient = kernels_gradient / batch_size
        return kernels_gradient

    def optimize(self, gradient):
        self.optimizer.optimize(self.kernels, gradient)

        avg = np.average(np.average(np.average(self.dloss, axis=2), axis=2), axis=0, keepdims=True)
        self.biases += -self.optimizer.learning_rate * avg.transpose()


class Pooling(Layer):
    def __init__(self, size, filter_size=(2, 2), pooling_type="max"):
        super(Pooling, self).__init__()
        self.size = size
        self.img_shape = size[-2:]
        self.pooling_type = pooling_type
        self.pooling_f = None
        self.filter_size = filter_size
        self.v_stride = filter_size[0]
        self.h_stride = filter_size[1]

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

        self.mask = None

        if len(size) != 3:
            raise ValueError("size parameter should be a tuple (k, m, n) where k indicates the number of channels and "
                             "m, n indicate the shape of the image")

        if (self.img_shape[1] - self.filter_size[1]) % self.h_stride != 0 or \
                (self.img_shape[0] - self.filter_size[0]) % self.v_stride != 0:
            raise ValueError("selected filter cannot be used with this image size")

        if self.pooling_type == "max":
            self.pooling_f = np.max
        else:
            raise ValueError(f"unknown pooling type")

    def forward(self, x):
        pool = self.pooling_f(x[:, :, self.row_index, self.col_index], axis=2)

        if self.pooling_type == "max":
            x[:, :, self.row_index, self.col_index] = (x[:, :, self.row_index, self.col_index] == pool[:, :, np.newaxis, :]) * 1
            self.mask = x

        pool = pool.reshape((x.shape[0], x.shape[1],
                             (self.img_shape[0] - self.filter_size[0]) // self.v_stride + 1,
                             (self.img_shape[1] - self.filter_size[1]) // self.h_stride + 1))

        self.out = pool
        return pool

    def backward(self, dloss):
        dloss = dloss.reshape(dloss.shape[0], dloss.shape[1], 1, -1)
        z = np.zeros_like(self.mask)
        z[:, :, self.row_index, self.col_index] = self.mask[:, :, self.row_index, self.col_index] * dloss
        return z

    def gradient(self, dloss, out, batch_size):
        return None

    def optimize(self, gradient):
        pass


class Flatten(Layer):
    def __init__(self, size):
        super(Flatten, self).__init__()
        self.size = size
        self.batch_size = None

    def forward(self, x):
        self.batch_size = x.shape[0]
        x = x.reshape((x.shape[0], -1))
        self.out = x
        return x

    def backward(self, dloss):
        return dloss.reshape((self.batch_size, self.size[0], self.size[1], self.size[2]))

    def gradient(self, dloss, out, batch_size):
        return None

    def optimize(self, gradient):
        pass


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.out = np.maximum(0, x)
        return self.out

    def backward(self, dloss):
        return (self.out > 0) * dloss


class SoftMax(Layer):
    def __init__(self):
        super().__init__()

    def softmax(self, x):
        x_scaled = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x_scaled)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, x):
        self.out = self.softmax(x)
        return self.out

    def backward(self, dloss):
        sm = self.softmax(self.out)
        jac = np.eye(sm.shape[1], sm.shape[1])[np.newaxis, :, :]
        jac = jac - sm[:, np.newaxis, :]
        jac = jac * sm[:, :, np.newaxis]
        return np.sum(jac * dloss[:, :, np.newaxis], axis=1)


class BatchNorm(Layer):
    def __init__(self):
        super().__init__()

        self.x = None
        self.m = None
        self.n = None
        self.initial_shape = None
        self.mean = None
        self.std = None

        self.beta = np.array([[0]], dtype=np.float64)
        self.gamma = np.array([[1]], dtype=np.float64)
        self.normalized = None
        self.epsilon = 1e-8

        self.mean_ma = 0
        self.var_ma = 0
        self.momentum = 0.9

        self.learning_rate = 0.001

    def forward(self, x):
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

    def backward(self, dloss):
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

    def inference(self, x):
        m = x.shape[0]
        n = np.prod(x.shape[1:])
        initial_shape = x.shape
        if len(x.shape) > 2:
            x = x.reshape((m, n))
        normalized = (x - self.mean_ma) / np.sqrt((self.var_ma + self.epsilon))
        transormed = self.gamma * normalized + self.beta
        transormed = transormed.reshape(initial_shape)
        return transormed
