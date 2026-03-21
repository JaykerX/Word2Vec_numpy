import numpy as np
import random
from collections import Counter
import os
import zipfile
import urllib.request

def download_dataset(url, path, inner_file):
    if not os.path.exists(path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, path)

    with zipfile.ZipFile(path) as z:
        text = z.read(inner_file).decode("utf-8")

    return text

def tokenize(text):
    return text.split()

def build_vocab(tokens, vocab_size=30000):
    counts = Counter(tokens)
    most_common = counts.most_common(vocab_size)

    vocab = {w: i for i, (w, _) in enumerate(most_common)}
    id2word = {i: w for w, i in vocab.items()}

    word_counts = np.array([counts[w] for w, _ in most_common])
    return vocab, id2word, word_counts

def subsample(tokens, word_counts, vocab, t=1e-5):
    total = sum(word_counts)
    freqs = word_counts / total
    prob_drop = 1 - np.sqrt(t / freqs)

    result = []
    for w in tokens:
        if w in vocab:
            i = vocab[w]
            if random.random() > prob_drop[i]:
                result.append(w)
    return result


def generate_pairs_stream(tokens, vocab, window_size):
    for i in range(len(tokens)):
        if tokens[i] not in vocab:
            continue

        center = vocab[tokens[i]]

        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j and tokens[j] in vocab:
                yield center, vocab[tokens[j]]


def get_negative_distribution(word_counts):
    p = word_counts ** 0.75
    return p / np.sum(p)


class Word2VecSGNS:
    def __init__(self, vocab_size, dim):
        self.W = np.random.randn(vocab_size, dim) * 0.01
        self.W_out = np.random.randn(vocab_size, dim) * 0.01

    def sigmoid(self, x):
        pos_mask = (x >= 0)
        neg_mask = ~pos_mask
        z = np.zeros_like(x)
        z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        z[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
        return z

    def train_step(self, center, context, negatives, lr):
        v_c = self.W[center].copy()
        v_o = self.W_out[context].copy()
        v_neg = self.W_out[negatives].copy()

        score_pos = np.dot(v_c, v_o)
        score_neg = v_neg @ v_c

        sig_pos = self.sigmoid(score_pos)
        sig_neg = self.sigmoid(score_neg)

        grad_vc = (sig_pos - 1) * v_o + (sig_neg @ v_neg) / len(negatives)
        grad_vo = (sig_pos - 1) * v_c
        grad_vnegs = np.outer(sig_neg, v_c)

        grad_vc   = np.clip(grad_vc,   -5, 5)
        grad_vo   = np.clip(grad_vo,   -5, 5)
        grad_vnegs = np.clip(grad_vnegs, -5, 5)

        self.W[center]      -= lr * grad_vc
        self.W_out[context] -= lr * grad_vo

        np.add.at(self.W_out, negatives, -lr * grad_vnegs)

        loss = -np.log(sig_pos + 1e-10) - np.sum(np.log(self.sigmoid(-score_neg) + 1e-10))
        return loss


def train(vocab_size, dim, window_size, neg_samples, lr, lr_min, epochs, train_tokens, url="http://mattmahoney.net/dc/text8.zip", path="text8.zip", inner_file="text8"):
    text = download_dataset(url, path, inner_file)
    tokens = tokenize(text)

    print("Building vocab...")
    tokens = tokens[:train_tokens]
    vocab, id2word, word_counts = build_vocab(tokens, vocab_size=vocab_size)

    tokens = [w for w in tokens if w in vocab]
    tokens = subsample(tokens, word_counts, vocab)

    neg_dist = get_negative_distribution(word_counts)

    model = Word2VecSGNS(len(vocab), dim=dim)

    window_size = window_size
    neg_samples = neg_samples
    lr = lr
    epochs = epochs
    total_steps = len(tokens) * epochs
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0
        count = 0

        for center, context in generate_pairs_stream(tokens, vocab, window_size):
            negatives = np.random.choice(len(vocab), size=neg_samples, p=neg_dist)
            lr_current = lr * max(1 - global_step / total_steps, lr_min)
            loss = model.train_step(center, context, negatives, lr_current)

            global_step += 1
            loss = model.train_step(center, context, negatives, lr_current)

            total_loss += loss
            count += 1

            if count % 100000 == 0:
                print(f"Step {count}, Avg Loss: {total_loss / count:.4f}")

        print(f"Epoch {epoch+1} done, Avg Loss: {total_loss / count:.4f}")

    return model, vocab, id2word


if __name__ == "__main__":

    model, vocab, id2word = train(vocab_size=30000, dim=100, window_size=2, neg_samples=5, lr=0.025, lr_min=0.0001, epochs=2, train_tokens=200000)

    path = "word2vec_model_big.npz"

    np.savez(
        path,
        W=model.W,
        W_out=model.W_out,
        vocab=vocab,
        id2word=id2word
    )

    print(f"Model saved to {path}")