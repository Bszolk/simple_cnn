import numpy as np
import pandas as pd
from model import Model
from layers import Linear, Input, ReLU, SoftMax


fac = 0.99 / 255

train = None
train_files = ["data/mnist_train1.csv", "data/mnist_train2.csv", "data/mnist_train3.csv", "data/mnist_train4.csv", "data/mnist_train5.csv", "data/mnist_train6.csv"]
for file in train_files:
    batch = np.loadtxt(file, delimiter=',')
    if train is not None:
        train = np.concatenate([train, batch])
    else:
        train = batch

X = train[:, 1:] * fac + 0.01
y = train[:, 0]
yOH = pd.get_dummies(y, dtype=int).to_numpy()

X_train = X[:50000]
y_trainOH = yOH[:50000]
X_valid = X[50000:]
y_validOH = yOH[50000:]

test = np.loadtxt("data/mnist_test.csv", delimiter=',')
X_test = test[:, 1:] * fac + 0.01
y_test = test[:, 0]
y_testOH = pd.get_dummies(y_test, dtype=int).to_numpy()

model = Model.sequential(
    Input(784),
    Linear(512),
    ReLU(),
    Linear(256),
    ReLU(),
    Linear(10),
    SoftMax()
)
model.set_loss_function("CrossEntropy")
model.set_batch_size(128)
model.set_optimizer("adam", 0.001)

model.train(X_train, y_trainOH, X_valid, y_validOH, epochs=2)

model.valid(X_test, y_validOH)

