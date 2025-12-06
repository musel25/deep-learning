# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#     path: /home/sylvain/.local/share/jupyter/kernels/python3
# ---

# %% [markdown]
# # Word embedding and RNN for sentiment analysis
#
# The goal of the following notebook is to predict whether a written
# critic about a movie is positive or negative. For that we will try three
# models. A simple linear model on the word embeddings, a recurrent neural
# network and a CNN.
#
# ## Preliminaries
#
# ### Libraries and Imports
#
# First some imports are needed.

# %%
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

# %% [markdown]
# ### Global variables
#
# First let’s define a few variables. `EMBEDDING_DIM` is the dimension of
# the vector space used to embed all the words of the vocabulary.
# `SEQ_LENGTH` is the maximum length of a sequence, `BATCH_SIZE` is the
# size of the batches used in stochastic optimization algorithms and
# `NUM_EPOCHS` the number of times we are going thought the entire
# training set during the training phase.

# %%
# <answer>
EMBEDDING_DIM = 8
SEQ_LENGTH = 64
BATCH_SIZE = 512
NUM_EPOCHS = 10
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# </answer>

# %% [markdown]
# ## The `IMDb` dataset
#
# We use the `datasets` library to load the `IMDb` dataset.

# %%
dataset = load_dataset("imdb")
train_set = dataset['train']
test_set = dataset['test']

train_set[0]

print(f"Number of training examples: {len(train_set)}")
print(f"Number of testing examples: {len(test_set)}")

# %% [markdown]
# ### Building a vocabulary out of `IMDb` from a tokenizer
#
# We first need a tokenizer that takes a text a returns a list of tokens.
# There are many tokenizers available from other libraries. Here we use
# the `tokenizers` library.

# %%
# Use a word-level tokenizer in lower case
tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Lowercase()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# %% [markdown]
# Then we need to define the set of words that will be understood by the
# model: this is the vocabulary. We build it from the training set.

# %%
train_texts = train_set['text']
test_texts = test_set['text']

trainer = trainers.WordLevelTrainer(vocab_size=10000, special_tokens=["[UNK]", "[PAD]"])
tokenizer.train_from_iterator(train_texts, trainer)

vocab = tokenizer.get_vocab()

UNK_IDX, PAD_IDX = vocab["[UNK]"], vocab["[PAD]"]
VOCAB_SIZE = len(vocab)

tokenizer.encode("All your base are belong to us").tokens
tokenizer.encode("All your base are belong to us").ids

vocab['plenty']


# %% [markdown]
# ## The training loop
#
# The training loop is decomposed into 3 different functions:
#
# -   `train_epoch`
# -   `evaluate`
# -   `train`
#
# ### Collate function
#
# The collate function maps raw samples coming from the dataset to padded
# tensors of numericalized tokens ready to be fed to the model.

# %%
def collate_fn(batch):
    def collate(text):
        """Turn a text into a tensor of integers."""
        ids = tokenizer.encode(text).ids[:SEQ_LENGTH]
        return torch.LongTensor(ids)

    src_batch = [collate(sample["text"]) for sample in batch]

    # Pad list of tensors using `pad_sequence`
    # <answer>
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    # </answer>

    # Define the labels tensor
    # <answer>
    tgt_batch = torch.Tensor([sample["label"] for sample in batch])
    # </answer>

    return src_batch, tgt_batch


# %% [markdown]
# ### The `accuracy` function
#
# We need to implement an accuracy function to be used in the
# `train_epoch` function (see below).

# %%
def accuracy(predictions, labels):
    # `predictions` and `labels` are both tensors of same length

    # Implement accuracy
    # <answer>
    return torch.sum((torch.sigmoid(predictions) > 0.5).float() == (labels > .5)).item() / len(
        predictions
    )
    # </answer>

assert accuracy(torch.Tensor([1, -2, 3]), torch.Tensor([1, 0, 1])) == 1
assert accuracy(torch.Tensor([1, -2, -3]), torch.Tensor([1, 0, 1])) == 2 / 3


# %% [markdown]
# ### The `train_epoch` function

# %%
def train_epoch(model: nn.Module, optimizer: Optimizer):
    model.to(DEVICE)

    # Training mode
    model.train()

    loss_fn = nn.BCEWithLogitsLoss()

    train_dataloader = DataLoader(
        train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )

    matches = 0
    losses = 0
    for sequences, labels in train_dataloader:
        sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)

        # Implement a step of the algorithm:
        #
        # - set gradients to zero
        # - forward propagate examples in `batch`
        # - compute `loss` with chosen criterion
        # - back-propagate gradients
        # - gradient step
        # <answer>
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        losses += loss.item()
        # </answer>

        acc = accuracy(predictions, labels)

        matches += len(predictions) * acc

    return losses / len(train_set), matches / len(train_set)


# %% [markdown]
# ### The `evaluate` function

# %%
def evaluate(model: nn.Module):
    model.to(DEVICE)
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()

    val_dataloader = DataLoader(
        test_set, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    losses = 0
    matches = 0
    for sequences, labels in val_dataloader:
        sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)

        predictions = model(sequences)
        loss = loss_fn(predictions, labels)
        acc = accuracy(predictions, labels)
        matches += len(predictions) * acc
        losses += loss.item()

    return losses / len(test_set), matches / len(test_set)


# %% [markdown]
# ### The `train` function

# %%
def train(model, optimizer):
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss, train_acc = train_epoch(model, optimizer)
        end_time = timer()
        val_loss, val_acc = evaluate(model)
        print(
            f"Epoch: {epoch}, "
            f"Train loss: {train_loss:.3f}, "
            f"Train acc: {train_acc:.3f}, "
            f"Val loss: {val_loss:.3f}, "
            f"Val acc: {val_acc:.3f}, "
            f"Epoch time = {(end_time - start_time):.3f}s"
        )


# %% [markdown]
# ### Helper function to predict from a character string

# %%
def predict_sentiment(model, sentence):
    "Predict sentiment of given sentence according to model"

    tensor, _ = collate_fn([{"label": 0, "text": sentence}])
    model.to(DEVICE)
    tensor = tensor.to(DEVICE)
    prediction = model(tensor)
    pred = torch.sigmoid(prediction)
    return pred.item()


# %% [markdown]
# ## Models
#
# ### Training a linear classifier with an embedding
#
# We first test a simple linear classifier on the word embeddings.

# %%
class EmbeddingNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Define an embedding of `vocab_size` words into a vector space
        # of dimension `embedding_dim`.
        # <answer>
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        # </answer>

        # Define a linear layer from dimension `seq_length` *
        # `embedding_dim` to 1.
        # <answer>
        self.l1 = nn.Linear(self.seq_length * self.embedding_dim, 1)
        # </answer>

    def forward(self, x):
        # `x` is of size `seq_length` * `batch_size`

        # Compute the embedding `embedded` of the batch `x`. `embedded` is
        # of size `seq_length` * `batch_size` * `embedding_dim`
        # <answer>
        embedded = self.embedding(x)
        # </answer>

        # Flatten the embedded words and feed it to the linear layer. `flatten`
        # must be of size `batch_size` * (`seq_length` * `embedding_dim`). You
        # might need to use `permute` first.
        # <answer>
        flatten = embedded.permute((1, 0, 2)).reshape(-1, self.seq_length * self.embedding_dim)
        # </answer>

        # Apply the linear layer and return a squeezed version
        # `l1` is of size `batch_size`
        # <answer>
        return self.l1(flatten).squeeze()
        # </answer>


# %%
embedding_net = EmbeddingNet(VOCAB_SIZE, EMBEDDING_DIM, SEQ_LENGTH)
print(sum(torch.numel(e) for e in embedding_net.parameters() if e.requires_grad))

print(
    VOCAB_SIZE * EMBEDDING_DIM + # Embeddings
    (SEQ_LENGTH * EMBEDDING_DIM + 1) # Linear
)

optimizer = Adam(embedding_net.parameters())
train(embedding_net, optimizer)

# %% [markdown]
# ### Training a linear classifier with a pretrained embedding
#
# Load a GloVe pretrained embedding instead

# %%
# Download GloVe word embedding
import gensim.downloader
glove_vectors = gensim.downloader.load('glove-twitter-25')

unknown_vector = glove_vectors.get_mean_vector(glove_vectors.index_to_key)
vocab_vectors = torch.tensor(np.stack([glove_vectors[e] if e in glove_vectors else unknown_vector for e in vocab.keys()]))

class GloVeEmbeddingNet(nn.Module):
    def __init__(self, seq_length, vocab_vectors, freeze=True):
        super().__init__()
        self.seq_length = seq_length

        # Define `embedding_dim` from vocabulary and the pretrained `embedding`.
        # <answer>
        self.embedding_dim = vocab_vectors.size(1)
        self.embedding = nn.Embedding.from_pretrained(vocab_vectors, freeze=freeze)
        # </answer>

        self.l1 = nn.Linear(self.seq_length * self.embedding_dim, 1)

    def forward(self, x):
        # Same forward as in `EmbeddingNet`
        # `x` is of size `batch_size` * `seq_length`
        # <answer>

        # `embedded` is of size `seq_length` * `batch_size` * `embedding_dim`
        embedded = self.embedding(x)

        # `flatten` is of size `batch_size` * `(seq_length * embedding_dim)`
        flatten = embedded.permute((1, 0, 2)).reshape(-1, self.seq_length * self.embedding_dim)
        # </answer>

        # L1 is of size batch_size
        # <answer>
        return self.l1(flatten).squeeze()
        # </answer>


# %% [markdown]
# ### Use pretrained embedding without fine-tuning

# %%
glove_embedding_net_freeze = GloVeEmbeddingNet(SEQ_LENGTH, vocab_vectors, freeze=True)
print(sum(torch.numel(e) for e in glove_embedding_net_freeze.parameters() if e.requires_grad))

print(
    (SEQ_LENGTH * 25 + 1) # Linear
)

optimizer = Adam(glove_embedding_net_freeze.parameters())
train(glove_embedding_net_freeze, optimizer)

# %% [markdown]
# ### Fine-tuning the pretrained embedding

# %%
# Define model and don't freeze embedding weights
# <answer>
glove_embedding_net = GloVeEmbeddingNet(SEQ_LENGTH, vocab_vectors, freeze=False)
# </answer>

# %% [markdown]
# ### Recurrent neural network with frozen pretrained embedding

# %%
class RNN(nn.Module):
    def __init__(self, hidden_size, vocab_vectors, freeze=True):
        super(RNN, self).__init__()

        # Define pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab_vectors, freeze=freeze)

        # Size of input `x_t` from `embedding`
        self.embedding_size = self.embedding.embedding_dim
        self.input_size = self.embedding_size

        # Size of hidden state `h_t`
        self.hidden_size = hidden_size

        # Define a GRU
        # <answer>
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size)
        # </answer>

        # Linear layer on last hidden state
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None):
        # `x` is of size `seq_length` * `batch_size` and `h0` is of size 1
        # * `batch_size` * `hidden_size`

        # Define first hidden state in not provided
        if h0 is None:
            # Get batch and define `h0` which is of size 1 * `batch_size` *
            # `hidden_size`
            # <answer>
            batch_size = x.size(1)
            h0 = torch.zeros(1, batch_size, self.hidden_size).to(DEVICE)
            # </answer>

        # `embedded` is of size `seq_length` * `batch_size` *
        # `embedding_dim`
        embedded = self.embedding(x)

        # Define `output` and `hidden` returned by GRU:
        #
        # - `output` is of size `seq_length` * `batch_size` * `embedding_dim`
        #   and gathers all the hidden states along the sequence.
        # - `hidden` is of size 1 * `batch_size` * `embedding_dim` and is the
        #   last hidden state.
        # <answer>
        output, hidden = self.gru(embedded, h0)
        # </answer>

        # Apply a linear layer on the last hidden state to have a score tensor
        # of size 1 * `batch_size` * 1, and return a one-dimensional tensor of
        # size `batch_size`.
        # <answer>
        return self.linear(hidden).squeeze()
        # </answer>


rnn = RNN(hidden_size=100, vocab_vectors=vocab_vectors)
print(sum(torch.numel(e) for e in rnn.parameters() if e.requires_grad))

hidden_size = 100
print(
    3 * hidden_size * (hidden_size + 25 + 2) + # GRU (2 bias vectors instead of 1)
    hidden_size + 1 # Linear
)

optimizer = optim.Adam(rnn.parameters(), lr=0.005)
train(rnn, optimizer)
