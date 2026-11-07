import numpy as np

def softmax_loss_naive(W, X, y, reg):

    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        loss += -np.log(probs[y[i]])

        for j in range(num_classes):
            dW[:, j] += (probs[j] - (j == y[i])) * X[i]

    loss /= num_train
    dW   /= num_train

    loss += 0.5 * reg * np.sum(W * W)
    dW   += reg * W

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Векторизованная версия softmax loss.
    W: (D, C), X: (N, D), y: (N,), reg: float
    """
    num_train = X.shape[0]

    # 1) Скоры (N, C)
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)   # стабилизация

    # 2) Вероятности (N, C)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 3) Loss: -log p[y_i], усреднённый по батчу
    correct_logprobs = -np.log(probs[np.arange(num_train), y])
    loss = np.mean(correct_logprobs)
    loss += 0.5 * reg * np.sum(W * W)

    # 4) Градиент
    dscores = probs.copy()                  # dL_i/ds_j = p_j
    dscores[np.arange(num_train), y] -= 1   # ... минус 1 для истинного класса
    dW = X.T.dot(dscores) / num_train
    dW += reg * W

    return loss, dW
