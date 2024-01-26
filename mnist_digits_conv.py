import numpy as np
import pandas as pd
from model import Model
from layers import Linear, Input, Convolutional, Pooling, Flatten, ReLU, SoftMax, BatchNorm

fac = 0.99 / 255

train = None
train_files = ["data/mnist_train1.csv", "data/mnist_train2.csv", "data/mnist_train3.csv", "data/mnist_train4.csv", "data/mnist_train5.csv", "data/mnist_train6.csv"]
for file in train_files:
    batch = np.loadtxt(file, delimiter=',')
    if train is not None:
        train = np.concatenate([train, batch])
    else:
        train = batch

X_train = train[:, 1:] * fac + 0.01
y_train = train[:, 0]
y_trainOH = pd.get_dummies(y_train, dtype=int).to_numpy()
X_train = X_train.reshape((X_train.shape[0], 28, 28))[:, np.newaxis, :, :]

test = np.loadtxt("data/mnist_test.csv", delimiter=',')
X_test = test[:, 1:] * fac + 0.01
y_test = test[:, 0]
y_testOH = pd.get_dummies(y_test, dtype=int).to_numpy()
X_test = X_test.reshape((X_test.shape[0], 28, 28))[:, np.newaxis, :, :]

model = Model.create(
    Input((1, 28, 28)),
    Convolutional((1, 28, 28), (1, 3, 3), "valid", 1, 4),
    ReLU(),
    BatchNorm(),
    Pooling((4, 26, 26), (2, 2), "max"),
    Flatten((4, 13, 13)),
    Linear(256),
    ReLU(),
    Linear(64),
    ReLU(),
    Linear(10),
    SoftMax()
)

model.set_loss_function("categorical_cross_entropy")
model.set_batch_size(128)
model.set_optimizer("adam", 0.001)

model.fit(X_train, y_trainOH, epochs=150)

model.validate(X_train[:10000], y_trainOH[:10000])
model.validate(X_test, y_testOH)


