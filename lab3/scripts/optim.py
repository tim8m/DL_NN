import numpy as np


def sgd(w, dw, config=None):
    config = {} if config is None else config
    return w - config.setdefault('learning_rate', 1e-2) * dw, config


def sgd_momentum(w, dw, config=None):
    config = {} if config is None else config
    lr = config.setdefault('learning_rate', 1e-2)
    momentum = config.setdefault('momentum', 0.9)
    v = momentum * config.get('velocity', np.zeros_like(w)) - lr * dw
    config['velocity'] = v
    return w + v, config


def rmsprop(w, dw, config=None):
    config = {} if config is None else config
    lr = config.setdefault('learning_rate', 1e-2)
    decay = config.setdefault('decay_rate', 0.99)
    eps = config.setdefault('epsilon', 1e-8)
    cache = decay * config.get('cache', np.zeros_like(w)) + (1 - decay) * dw**2
    config['cache'] = cache
    return w - lr * dw / (np.sqrt(cache) + eps), config


def adam(w, dw, config=None):
    config = {} if config is None else config
    lr = config.setdefault('learning_rate', 1e-3)
    beta1 = config.setdefault('beta1', 0.9)
    beta2 = config.setdefault('beta2', 0.999)
    eps = config.setdefault('epsilon', 1e-8)
    t = config.get('t', 0) + 1
    m = beta1 * config.get('m', np.zeros_like(w)) + (1 - beta1) * dw
    v = beta2 * config.get('v', np.zeros_like(w)) + (1 - beta2) * dw**2
    config.update(m=m, v=v, t=t)
    return w - lr * (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps), config
