import numpy as np


def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    gradient = np.zeros_like(W)
    for sample, label in zip(X, y):
        scores = sample @ W
        scores -= scores.max()
        exp_scores = np.exp(scores)
        total = exp_scores.sum()
        loss += np.log(total) - scores[label]
        for j, probability in enumerate(exp_scores / total):
            gradient[:, j] += sample * (probability - int(j == label))
    return loss / len(X) + reg * np.sum(W ** 2), gradient / len(X) + 2 * reg * W


def softmax_loss_vectorized(W, X, y, reg):
    scores = X @ W
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    totals = exp_scores.sum(axis=1, keepdims=True)
    rows = np.arange(len(X))
    loss = np.mean(np.log(totals[:, 0]) - scores[rows, y])
    probabilities = exp_scores / totals
    probabilities[rows, y] -= 1
    gradient = X.T @ probabilities / len(X)
    return loss + reg * np.sum(W ** 2), gradient + 2 * reg * W
