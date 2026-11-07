import numpy as np


def affine_forward(x, w, b):
    return x.reshape(len(x), -1) @ w + b, (x, w)


def affine_backward(dout, cache):
    x, w = cache
    return (dout @ w.T).reshape(x.shape), x.reshape(len(x), -1).T @ dout, dout.sum(axis=0)


def relu_forward(x):
    return np.maximum(x, 0), x


def relu_backward(dout, cache):
    return dout * (cache > 0)


def softmax_loss(scores, y):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    total = exp.sum(axis=1, keepdims=True)
    rows = np.arange(len(y))
    loss = np.mean(np.log(total[:, 0]) - shifted[rows, y])
    grad = exp / total
    grad[rows, y] -= 1
    return loss, grad / len(y)


def batchnorm_forward(x, gamma, beta, bn_param):
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    running_mean = bn_param.get('running_mean', np.zeros(x.shape[1], dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(x.shape[1], dtype=x.dtype))
    if bn_param['mode'] == 'test':
        return gamma * (x - running_mean) / np.sqrt(running_var + eps) + beta, None
    mean, var = x.mean(axis=0), x.var(axis=0)
    inv_std = 1 / np.sqrt(var + eps)
    normalized = (x - mean) * inv_std
    bn_param['running_mean'] = momentum * running_mean + (1 - momentum) * mean
    bn_param['running_var'] = momentum * running_var + (1 - momentum) * var
    return gamma * normalized + beta, (normalized, gamma, inv_std)


def batchnorm_backward(dout, cache):
    normalized, gamma, inv_std = cache
    dnorm = dout * gamma
    dx = inv_std * (dnorm - dnorm.mean(axis=0) -
                    normalized * (dnorm * normalized).mean(axis=0))
    return dx, (dout * normalized).sum(axis=0), dout.sum(axis=0)


def dropout_forward(x, dropout_param):
    mode, p = dropout_param['mode'], dropout_param['p']
    if mode == 'test':
        return x, (mode, None)
    seed = dropout_param.get('seed')
    rng = np.random if seed is None else np.random.RandomState(seed)
    mask = (rng.random(x.shape) < p).astype(x.dtype) / p
    return x * mask, (mode, mask)


def dropout_backward(dout, cache):
    mode, mask = cache
    return dout * mask if mode == 'train' else dout


def conv_forward_naive(x, w, b, conv_param):
    stride, pad = conv_param['stride'], conv_param['pad']
    n, _, h, width = x.shape
    f, _, kh, kw = w.shape
    oh, ow = 1 + (h + 2 * pad - kh) // stride, 1 + (width + 2 * pad - kw) // stride
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.empty((n, f, oh, ow), dtype=np.result_type(x, w, b))
    for i in range(oh):
        for j in range(ow):
            patch = padded[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]
            for k in range(f):
                out[:, k, i, j] = (patch * w[k]).sum(axis=(1, 2, 3)) + b[k]
    return out, (x, w, stride, pad, padded)


def conv_backward_naive(dout, cache):
    x, w, stride, pad, padded = cache
    _, _, kh, kw = w.shape
    dx = np.zeros_like(padded)
    dw = np.zeros_like(w)
    for i in range(dout.shape[2]):
        for j in range(dout.shape[3]):
            region = (slice(None), slice(None), slice(i*stride, i*stride+kh),
                      slice(j*stride, j*stride+kw))
            for k in range(w.shape[0]):
                grad = dout[:, k, i, j, None, None, None]
                dw[k] += (padded[region] * grad).sum(axis=0)
                dx[region] += w[k] * grad
    return dx[:, :, pad:pad+x.shape[2], pad:pad+x.shape[3]], dw, dout.sum(axis=(0, 2, 3))


def max_pool_forward_naive(x, pool_param):
    ph, pw, stride = (pool_param[k] for k in ('pool_height', 'pool_width', 'stride'))
    n, c, h, w = x.shape
    out = np.empty((n, c, 1+(h-ph)//stride, 1+(w-pw)//stride), dtype=x.dtype)
    for i in range(out.shape[2]):
        for j in range(out.shape[3]):
            out[:, :, i, j] = x[:, :, i*stride:i*stride+ph, j*stride:j*stride+pw].max(axis=(2, 3))
    return out, (x, ph, pw, stride)


def max_pool_backward_naive(dout, cache):
    x, ph, pw, stride = cache
    dx = np.zeros_like(x)
    n, c = np.indices(x.shape[:2])
    for i in range(dout.shape[2]):
        for j in range(dout.shape[3]):
            patch = x[:, :, i*stride:i*stride+ph, j*stride:j*stride+pw]
            arg = patch.reshape(*x.shape[:2], -1).argmax(axis=-1)
            dx[n, c, i*stride+arg//pw, j*stride+arg%pw] += dout[:, :, i, j]
    return dx
