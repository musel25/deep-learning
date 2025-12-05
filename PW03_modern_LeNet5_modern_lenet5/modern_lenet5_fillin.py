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
# # Implementing a modern LeNet5
#
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# %% [markdown]
# ## Load and prepare the MNIST dataset

# %%
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(int)

# %%
# Normalize and reshape
X = X / 255.0
X = X.reshape(-1, 1, 28, 28)  # shape: (n_samples, channels, height, width)

# %%
# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)

# %%
# Convert to Pytorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# %%
# Use dataloader to generate minibatches
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=1000, shuffle=False)


# %% [markdown]
# ## Define the modern LeNet-5 model

# %%
class ModernLeNet5(nn.Module):
    def __init__(self):
        super(ModernLeNet5, self).__init__()

        # Define a convolutional layer with 32 filters, 5x5 kernels. Adjust the
        # padding to maintain the height and width.
        self.conv1 = ...

        # Define a batch-normalization layer
        self.bn1 = ...

        # Define a convolutional layer with 64 filters, 5x5 kernels. Adjust the
        # padding to maintain the height and width.
        self.conv2 = ...

        # Define a batch-normalization layer
        self.bn2 = ...

        # Define a max-pooling step
        self.pool = ...

        # Define a dropout layer with dropout rate 0.25
        self.dropout = ...

        self.fc1 = ...
        self.bn3 = ...
        self.fc2 = ...

    def forward(self, x):
        # Apply a first convolutional layer, a batch-normalization, an
        # activation function and a pooling
        x = ...

        # Same transformation again
        x = ...

        # Use fully connected later, batch-normalization, activation function,
        # dropout and final fully connected layer
        x = ...
        x = ...
        x = ...

        return x


# %% [markdown]
# ## Training configuration

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModernLeNet5().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# %% [markdown]
# ## Train the model

# %%
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Set gradients to zero, forward propagate, compute loss, backward
        # propagate and update parameters
        ...

        running_loss += loss.item()

    print(f"Epoch [{epoch}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

# %% [markdown]
# ## Evaluate the model

# %%
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Predict classes of `images`
        ...

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# %% [markdown]
# ## Visualize predictions

# %%
model.eval()
all_images, all_preds, all_labels = [], [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)

        all_images.append(X_batch.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

# Stack all test data
X_all = np.concatenate(all_images)
y_all = np.concatenate(all_labels)
p_all = np.concatenate(all_preds)

# Find wrong predictions
wrong_idx = np.where(p_all != y_all)[0]

fig, axes = plt.subplots(5, 5, figsize=(8, 8))
fig.suptitle("MNIST Wrong Predictions", fontsize=14)

for ax, idx in zip(axes.flat, wrong_idx):
    img = X_all[idx][0]
    true_label = y_all[idx]
    pred_label = p_all[idx]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"T:{true_label} / P:{pred_label}", fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Bonus challenges
#
# 1.  Compute the fraction of total model parameters that belong to the
#     classification head (i.e., the fully connected layers) in your
#     current ModernLeNet5 architecture. Modify the model to replace the
#     flattening step with a Global Average Pooling (GAP) layer before the
#     classifier. What is the new fraction relative to the total parameter
#     count?
#
# 2.  Are the bias terms in a convolutional layer necessary when the layer
#     is immediately followed by a Batch Normalization layer?
