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
# # CBOW model trained on 20000 lieues sous les mers
#
# ## Needed libraries
#
# You will need the following new libraries:
#
# -   `spacy` for tokenizing
# -   `gensim` for cosine similarities (use `gensim>=4.0.0`)
#
# You will also need to download rules for tokenizing a french text.
#
# ``` bash
# python -m spacy download fr_core_news_sm
# ```

# %%
import numpy as np
import torch
from torch import nn
import torch.optim as optim

import spacy
from gensim.models.keyedvectors import KeyedVectors

# %% [markdown]
# ## Tokenizing the corpus

# %%
# Use a french tokenizer to create a tokenizer for the french language
spacy_fr = spacy.load("fr_core_news_sm")
with open("data/20_000_lieues_sous_les_mers.txt", "r", encoding="utf-8") as f:
    document = spacy_fr.tokenizer(f.read())

# Define a filtered set of tokens by iterating on `document`. Define a
# subset of tokens that are
#
# - alphanumeric
# - in lower case
tokens = ...

# Make a list of unique tokens and dictionary that maps tokens to
# their index in that list.
idx2tok = ...
tok2idx = ...


# %% [markdown]
# ## The continuous bag of words model

# %%
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # Define an Embedding module (`nn.Embedding`) and a linear
        # transform (`nn.Linear`) without bias.
        self.embeddings = ...
        self.U_transpose = ...

    def forward(self, context):
        # Implements the forward pass of the CBOW model
        # `context` is of size `batch_size` * NGRAMS

        # `e_i` is of size `batch_size` * NGRAMS * `embedding_size`
        e_i = ...

        # `e_bar` is of size `batch_size` * `embedding_size`
        e_bar = ...

        # `UT_e_bar` is of size `batch_size` * `vocab_size`
        UT_e_bar = ...

        return ...


# Set the size of vocabulary and size of embedding
VOCAB_SIZE = len(idx2tok)
EMBEDDING_SIZE = 32

# Create a Continuous bag of words model
cbow = CBOW(VOCAB_SIZE, EMBEDDING_SIZE)

# Send to GPU if any
device = "cuda:0" if torch.cuda.is_available() else "cpu"
cbow.to(device)


# %% [markdown]
# ## Preparing the data

# %%
# Generate n-grams for a given list of tokens, use yield, use window length of n-grams
def ngrams_iterator(token_list, ngrams):
    """Generates successive N-grams from a list of tokens."""

    for i in range(len(token_list) - ngrams + 1):
        idxs = [tok2idx[tok] for tok in token_list[i:i+ngrams]]

        # Get center element in `idxs`
        center = idxs.pop(ngrams // 2)

        # Yield the index of center word and indexes of context words
        # as a Numpy array (for Pytorch to automatically convert it to
        # a Tensor).
        yield center, np.array(idxs)


# Create center, context data
NGRAMS = 5
ngrams = list(ngrams_iterator(tokens, NGRAMS))

BATCH_SIZE = 512
data = torch.utils.data.DataLoader(ngrams, batch_size=BATCH_SIZE, shuffle=True)

# %% [markdown]
# ## Learn CBOW model

# %%
# Gradient descent algorithm to use
optimizer = ...

# Use a cross-entropy loss from the `nn` submodule
ce_loss = ...

# %%
EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for i, (center, context) in enumerate(data):
        center, context = center.to(device), context.to(device)

        # Reset the gradients of the computational graph
        ...

        # Forward pass
        UT_ebar = ...

        # Compute negative log-likelihood loss averaged over the
        # mini-batch
        loss = ...

        # Backward pass to compute gradients of each parameter
        ...

        # Gradient descent step according to the chosen optimizer
        ...

        total_loss += loss.data

        if i % 20 == 0:
            loss_avg = float(total_loss / (i + 1))
            print(
                f"Epoch ({epoch}/{EPOCHS}), batch: ({i}/{len(data)}), loss: {loss_avg}"
            )

    # Print average loss after each epoch
    loss_avg = float(total_loss / len(data))
    print("{}/{} loss {:.2f}".format(epoch, EPOCHS, loss_avg))

    # Predict if `predict_center_word` is implemented
    try:
        left_words = ["le", "capitaine"]
        right_words = ["me", "dit"]
        word = predict_center_word(word2vec, *left_words, *right_words)[0]
        print(" ".join(left_words + [word] + right_words))
    except NameError:
        pass


# %% [markdown]
# ## Prediction functions
#
# Now that the model is learned we can give it a context it has never seen
# and see what center word it predicts.

# %%
def predict_center_word_idx(cbow, *context_words_idx, k=10):
    """Return k-best center words given indexes of context words."""

    # Create a fake minibatch containing just one example
    fake_minibatch = ...

    # Forward propagate through the cbow model
    score_center = ...

    # Retrieve top k-best indexes using `torch.topk`
    _, best_idxs = ...

    # Return actual tokens using `idx2tok`
    return ...


def predict_center_word(cbow, *context_words, k=10):
    """Return k-best center words given context words."""

    idxs = [tok2idx[tok] for tok in context_words]
    return predict_center_word_idx(cbow, *idxs, k=k)


# %%
predict_center_word(cbow, "vingt", "mille", "sous", "les")
predict_center_word(cbow, "mille", "lieues", "les", "mers")
predict_center_word(cbow, "le", "capitaine", "fut", "le")
predict_center_word(cbow, "le", "commandant", "fut", "le")

# %% [markdown]
# ## Testing the embedding
#
# We use the library `gensim` to easily compute most similar words for the
# embedding we just learned. Use `gensim>=4.0.0`.

# %%
m = KeyedVectors(vector_size=EMBEDDING_SIZE)
m.add_vectors(idx2tok, cbow.embeddings.weight.detach().cpu().numpy())

# %% [markdown]
# You can now test most similar words for, for example “lieues”, “mers”,
# “professeur”… You can look at `words_decreasing_freq` to test most
# frequent tokens.

# %%
unique, freq = np.unique(tokens, return_counts=True)
idxs = freq.argsort()[::-1]
words_decreasing_freq = list(zip(unique[idxs], freq[idxs]))

# %%
.most_similar(...)
