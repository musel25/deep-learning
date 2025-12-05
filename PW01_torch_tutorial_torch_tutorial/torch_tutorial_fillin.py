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
# # Introduction to Pytorch
#
# ## `autodiff`
#
# ### Autodiff for simple gradient descent
#
# Load needed libraries
#
# $$
# \newcommand\p[1]{{\left(#1\right)}}
# \newcommand\code[1]{\texttt{#1}}
# $$

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# %% [markdown]
# Here is a simple example of how to find the minimum of the function
# $x\mapsto\p{x-3}^2$ using the autodiff functionality of Pytorch.
#
# First initialize a tensor `x` and indicate that we want to store a
# gradient on it.

# %%
x = torch.tensor([1.0], requires_grad=True)

# %% [markdown]
# Create an optimizer on parameters. Here we want to optimize w.r.t.
# variable `x`:

# %%
optimizer = optim.SGD([x], lr=0.01)

# %% [markdown]
# Create a computational graph using parameters (here only `x`) and
# potentially other tensors.
#
# Here we only want to compute $\p{x-3}^2$ so we define:

# %%
y = (x - 3) ** 2

# %% [markdown]
# Back-propagating gradients for `y` down to `x`. Don’t forget to reset
# gradients before.

# %%
optimizer.zero_grad()
y.backward()

# %% [markdown]
# Use gradient on `x` to apply a one-step gradient descent.

# %%
optimizer.step()
x.grad
x

# %% [markdown]
# And last we iterate the whole process

# %%
it = 0
while it < 1000:
    loss = (x - 3) ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 20 == 0:
        print("Iteration: %d, x: %f, loss: %f" % (it, x.item(), loss.item()))
    it += 1

# %% [markdown]
# ### Differentiate the exponential
#
# The exponential function can be approximated using its Taylor expansion:
# $$
# \exp\p{z}\approx\sum_{k=0}^{N}\frac{z^k}{k!}
# $$
#
# First define `x`, the “parameter” and build a computational graph from
# it to compute the exponential.

# %%
...

# %% [markdown]
# Compute the gradient and verify that it is correct

# %%
...

# %% [markdown]
# ### Solving equations with Pytorch
#
# Suppose we want to solve the following system of two equations
#
# $$
# e^{-e^{-(x_1 + x_2)}}=x_2 (1 + x_1^2)
# $$ $$
# x_1 \cos(x_2) + x_2 \sin(x_1)= 1/2
# $$
#
# Find a loss whose optimization leads to a solution of the system of
# equations above.

# %%
# Define two functions
...

# %% [markdown]
# Use Pytorch autodiff to solve the system of equations

# %%
...

# %% [markdown]
# ## Linear least squares in Pytorch
#
# ### Synthetic data
#
# We use the following linear model:
#
# $$
# y = \langle\beta,x\rangle+\varepsilon
# $$
#
# where $x\in\mathbb R^p$ and $\varepsilon\sim\mathcal N(0, \sigma^2)$.

# %%
import math

p = 512
N = 50000
X = torch.randn(N, p)
beta = torch.randn(p, 1) / math.sqrt(p)
y = torch.mm(X, beta) + 0.5 * torch.randn(N, 1)

# %% [markdown]
# ### Model implementation
#
# Every model in Pytorch is implemented as a class that derives from
# `nn.Module`. The two main methods to implement are:
#
# -   `__init__`: Declare needed building blocks to implement forward pass
# -   `forward`: Implement the forward pass from the input given as
#     argument

# %%
import torch.nn as nn


class LinearLeastSquare(nn.Module):
    def __init__(self, input_size):
        super(LinearLeastSquare, self).__init__()

        # Declaring neural networks building blocks. Here we only need
        # a linear transform.
        self.linear = ...

    def forward(self, input):
        # Implementing forward pass. Return corresponding output for
        # this neural network.
        return ...


# %% [markdown]
# ### Preparing dataset

# %%
from torch.utils.data import TensorDataset

# Gather data coming from Pytorch tensors using `TensorDataset`
dataset = ...

# %%
from torch.utils.data import DataLoader
# Define `train_loader` that is an iterable on mini-batches using
# `DataLoader`
batch_size = ...
train_loader = ...

# %%
# Loss function to use
from torch.nn import MSELoss
loss_fn = ...

# %%
# Optimization algorithm
from torch.optim import SGD

# Instantiate model with `LinearLeastSquare` with the correct input
# size.
model = ...

# %%
# Use the stochastic gradient descent algorithm with a learning rate of
# 0.01 and a momentum of 0.9.
optimizer = ...

# %% [markdown]
# ### Learning loop

# %%
epochs = 10
losses = []
for i in range(epochs):
    for src, tgt in train_loader:
        # Forward pass
        ...

        # Backpropagation on loss
        ...

        # Gradient descent step
        ...

        losses.append(loss.item())

    print(f"Epoch {i}/{epochs}: Last loss: {loss}")

# %%
x = np.arange(len(losses)) / len(losses) * epochs
plt.plot(x, losses)

# %% [markdown]
# From the model what should be the minimum MSE?
#
# …
#
# ### Learning loop with scheduler
#
# From convex optimization theory the learning rate should be decreasing
# toward 0. To have something approaching we use a scheduler that is
# updating the learning rate every epoch.

# %%
from torch.optim.lr_scheduler import MultiStepLR

# Define a scheduler
model = ...
optimizer = ...
scheduler = ...

# %%
# Implement the learning loop with a scheduler
...


# %% [markdown]
# ## Multi-layer perceptron
#
# Implement a multi-layer perceptron described by the following function:
# $$
# f\p{x,\beta}=W_3\sigma\p{W_2\sigma\p{W_1 x}}
# $$ where $\sigma\p{x}=\max\p{x, 0}$.

# %%
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MultiLayerPerceptron, self).__init__()

        # Define hyperparameters of neural network and building blocks
        ...

    def forward(self, x):
        # Implement forward pass
        ...


# %% [markdown]
# ### Synthetic 2-dimensional spiral dataset

# %%
n_classes = 3
n_loops = 2
n_samples = 1500

def spirals(n_classes=3, n_samples=1500, n_loops=2):
    klass = np.random.choice(n_classes, n_samples)
    radius = np.random.rand(n_samples)
    theta = klass * 2 * math.pi / n_classes + radius * 2 * math.pi * n_loops
    radius = radius + 0.05 * np.random.randn(n_samples)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta))).astype("float32"), klass

X_, y_ = spirals(n_samples=n_samples, n_classes=n_classes, n_loops=n_loops)
plt.scatter(X_[:, 0], X_[:, 1], c=y_)

# %% [markdown]
# ### Preparing dataset

# %%
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

X = torch.from_numpy(X_)
y = torch.from_numpy(y_)
dataset = TensorDataset(X, y)
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
X_, y_ = spirals(n_samples=1000, n_classes=n_classes, n_loops=n_loops)
X = torch.from_numpy(X_)
y = torch.from_numpy(y_)
test_set = TensorDataset(X, y)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# %% [markdown]
# ### The learning loop

# %%
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss

loss_fn = CrossEntropyLoss()
model = MultiLayerPerceptron(2, 20, 20, n_classes)
optimizer = SGD(model.parameters(), lr=0.05)
optimizer = Adam(model.parameters())

# %%
import copy

epochs = 1000
losses = []
models = []
for i in range(epochs):
    for src, tgt in train_loader:
        ...

    # Accuracy on the test set
    acc = 0.
    for src, tgt in test_loader:
        prd = model(src).detach().argmax(dim=1)
        acc += sum(prd == tgt).item()

    acc /= len(test_set)
    if i % 20 == 0:
        print(f"Epoch {i}/{epochs}: Test accuracy: {acc}")

    models.append(copy.deepcopy(model))


# %%
def get_image_data(model, colors, xs, ys):
    """Return color image of size H*W*4."""

    # Generate points in grid
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((xx.ravel(), yy.ravel())).astype("float32")
    points = torch.from_numpy(points)

    # Predict class probability on points
    prd = model(points).detach()
    prd = torch.nn.functional.softmax(prd, dim=1)

    # Build a color image from colors
    colors = torch.from_numpy(colors)
    img = torch.mm(prd, colors).numpy()
    img = img.reshape((ynum, xnum, 4))
    img = np.minimum(img, 1)

    return img

fig, ax = plt.subplots()

# Get n_classes colors in RGBa form
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
import matplotlib as mpl
colors = mpl.colors.to_rgba_array(colors)[:n_classes, :4].astype("float32")

# Draw scatter plot of test set using colors
ax.scatter(X[:, 0], X[:, 1], c=colors[y])
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
xnum, ynum = (int(i) for i in fig.dpi * fig.get_size_inches())

# Create discretization
xs = np.linspace(xmin, xmax, xnum)
ys = np.linspace(ymin, ymax, ynum)
img = get_image_data(model, colors, xs, ys)

ax.imshow(img, extent=[xmin, xmax, ymin, ymax], origin="lower", alpha=.7)
