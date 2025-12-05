# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#     path: /usr/share/jupyter/kernels/python3
# ---

# %% [markdown]
# # Neural networks from scratch
#
# ## Libraries and dataset

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

n_classes = 4
n_loops = 1
n_samples = 1500

def spirals(n_classes=3, n_samples=1500, n_loops=2):
    klass = np.random.choice(n_classes, n_samples)
    radius = np.random.rand(n_samples)
    theta = klass * 2 * math.pi / n_classes + radius * 2 * math.pi * n_loops
    radius = radius + 0.05 * np.random.randn(n_samples)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta))).astype("float32"), klass

X, y = spirals(n_samples=n_samples, n_classes=n_classes, n_loops=n_loops)

# %% [markdown]
# ## Visualize the dataset

# %%
...


# %% [markdown]
# ## Activation functions
#
# ReLU and sigmoid function and their derivative (should work for numpy
# array of any dimension (1D, 2D,…))

# %%
def relu(v):
    ...


def drelu(v):
    ...


def sigmoid(v):
    ...


def dsigmoid(v):
    ...


# %% [markdown]
# ## Define the Multilayer Perceptron
#
# First define the shape of the multilayer perceptron:
#
# -   `n0`: size of input,
# -   `n1`: size of hidden layer,
# -   `n2`: size of output.

# %%
n0 = ...
n1 = ...
n2 = ...

# %% [markdown]
# Variables for weights, biases of each layers and gradients of loss wrt
# to any intermediate quantity.

# %%
# Random weights
W1 = np.random.randn(n0, n1)
W2 = np.random.randn(n1, n2)

# Biases set to zero
b1 = np.zeros((n1, 1))
b2 = np.zeros((n2, 1))

# Gradients of loss
Lx_2 = np.zeros((n2, 1))
LW_2 = np.zeros((n1, n2))
Lb_2 = np.zeros((n2, 1))

Lx_1 = np.zeros((n1, 1))
LW_1 = np.zeros((n0, n1))
Lb_1 = np.zeros((n1, 1))

# %% [markdown]
# What about He’s initialization for `W1` and `W2`?

# %%
# Random weights with He's initialization
W1 = ...
W2 = ...

# %% [markdown]
# Define the learning rate and the activation functions along their
# derivatives at each layer:
#
# -   `eta`: learning rate
# -   `af`, `daf`: activation function and its derivative for hidden layer

# %%
# Define eta, af, daf
eta = ...
af = ...
daf = ...

# %% [markdown]
# ## The learning loop (no minibatch)

# %%
nepochs = 15
for epoch in range(nepochs + 1):
    acc_epoch = 0

    # Here we are using stochastic gradient descent (minibatch of size 1)
    for idx, (x0, y2) in enumerate(zip(X, y)):
        x0 = x0.reshape((-1, 1))

        # Implement the forward pass: use `W1`, `x0`, `b1`, `af`, `W2`, `x1`,
        # `b2` to define `z1`, `x1`, `z2`, `x2`. Remember that there is no
        # activation function for the last layer
        z1 = ...
        x1 = ...
        z2 = ...
        x2 = ...

        # Predicted class
        pred = np.argmax(x2)
        acc_epoch += (pred == y2)

        if idx % 100 == 0:
            print(f"Epoch: {epoch:02}, sample: {idx:04}, class: {y2}, pred: {pred}, output: {x2}")

        # One-hot encoding of class `y2`
        y2_one_hot = np.zeros((n2, 1))
        y2_one_hot[y2, 0] = 1

        # Softmax of output needed in the loss
        softmax_x2 = np.exp(x2) / sum(np.exp(x2))

        # Gradient of loss wrt output layer
        Lx_2 = ...

        # Gradient of loss wrt weights and biases in second layer
        LW_2 = ...
        Lb_2 = ...

        # Gradient of loss wrt first layer
        Lx_1 = ...

        # Gradient of loss wrt weights and biases in first layer
        LW_1 = ...
        Lb_1 = ...

        # Gradient descent step: use `eta`, `Lw_1` `Lw_2` `Lb_1` `Lb_2` to
        # update `W1`, `W2`, `b1`, `b2`.
        W1 -= ...
        W2 -= ...
        b1 -= ...
        b2 -= ...

    print(f"Epoch: {epoch:02}, training accuracy: {acc_epoch/n_samples}")

# %% [markdown]
# ## Visualization

# %%
num = 250
xx = np.linspace(X[:, 0].min(), X[:, 0].max(), num)
yy = np.linspace(X[:, 1].min(), X[:, 1].max(), num)
XX, YY = np.meshgrid(xx, yy)
points = np.c_[XX.ravel(), YY.ravel()]

# Forward pass on all points
z1 = W1.T @ points.T + b1
x1 = af(z1)
z2 = W2.T @ x1 + b2
x2_hat = np.argmax(z2, axis=0)

C = x2_hat.reshape(num, num)

plt.contourf(XX, YY, C, cmap=plt.get_cmap("tab10"), alpha=.4)
plt.scatter(*X.T, c=y, cmap=plt.get_cmap("tab10"))

plt.show()

# %% [markdown]
# ## The learning loop with minibatch

# %%
n0 = 2
n1 = 100
n2 = n_classes

nepochs = 1000
batch_size = 64

# Random weights
W1 = np.random.randn(n0, n1)
W2 = np.random.randn(n1, n2)

# Biases set to zero
b1 = np.zeros(n1)
b2 = np.zeros(n2)

# Gradients of loss
LX_2 = np.zeros((batch_size, n2))
LW_2 = np.zeros((batch_size, n1, n2))
Lb_2 = np.zeros((batch_size, n2))

LX_1 = np.zeros((batch_size, n1))
LW_1 = np.zeros((batch_size, n0, n1))
Lb_1 = np.zeros((batch_size, n1))


def fake_dataloader(X, y, batch_size=32, shuffle=True):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]


for epoch in range(nepochs + 1):
    acc_epoch = 0

    for idx, (X0, y2) in enumerate(fake_dataloader(X, y, batch_size=batch_size)):
        # Implement the forward pass: use `W1`, `X0`, `b1`, `af`, `W2`, `X1`,
        # `b2` to define `Z1`, `X1`, `Z2`, `X2`. This time, `X0` is batch_size * 2 !
        Z1 = ...
        X1 = ...
        Z2 = ...
        X2 = ...

        # Predicted class (use np.argmax with axis argument)
        pred = ...
        acc_epoch += sum(pred == y2)

        # One-hot encoding of classes in `y2` (use `np.eye`)
        y2_one_hot = ...

        # Softmax of output needed in the loss (use `np.sum` with `keepdims`)
        softmax_X2 = ...

        # Gradient of loss wrt output layer
        LX_2 = ...

        # Gradient of loss wrt weights and biases in second layer
        # Since `LW_2` is 3-dimensional, operator `@` is not working anymore.
        # Use `np.einsum` here.
        LW_2 = ...
        Lb_2 = ...

        # Gradient of loss wrt first layer
        # Use `np.einsum` again.
        LX_1 = ...

        # Gradient of loss wrt weights and biases in first layer
        LW_1 = ...
        Lb_1 = ...

        # Gradient descent step: use `eta`, `Lw_1` `Lw_2` `Lb_1` `Lb_2` to
        # update `W1`, `W2`, `b1`, `b2`. Don't forget to average gradients over
        # the minibatch.
        W1 -= ...
        W2 -= ...
        b1 -= ...
        b2 -= ...

    print(f"Epoch: {epoch:02}, training accuracy: {acc_epoch/n_samples}")

# %% [markdown]
# ## Visualization

# %%
num = 250
xx = np.linspace(X[:, 0].min(), X[:, 0].max(), num)
yy = np.linspace(X[:, 1].min(), X[:, 1].max(), num)
XX, YY = np.meshgrid(xx, yy)
points = np.c_[XX.ravel(), YY.ravel()]

# Forward pass on all points
Z1 = points @ W1 + b1
X1 = af(Z1)
Z2 = X1 @ W2 + b2
X2 = Z2
X2_hat = np.argmax(Z2, axis=1)

C = X2_hat.reshape(num, num)

plt.contourf(XX, YY, C, cmap=plt.get_cmap("tab10"), alpha=.4)
plt.scatter(*X.T, c=y, cmap=plt.get_cmap("tab10"))

plt.show()
