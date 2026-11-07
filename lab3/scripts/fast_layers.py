import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv_forward_fast(x, w, b, conv_param):
    stride, pad = conv_param['stride'], conv_param['pad']
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    windows = sliding_window_view(padded, w.shape[-2:], axis=(2, 3))[:, :, ::stride, ::stride]
    n, c, oh, ow, kh, kw = windows.shape
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n*oh*ow, c*kh*kw)
    out = cols @ w.reshape(len(w), -1).T + b
    return out.reshape(n, oh, ow, len(w)).transpose(0, 3, 1, 2), (x.shape, w, cols, stride, pad)


def conv_backward_fast(dout, cache):
    shape, w, cols, stride, pad = cache
    n, c, h, width = shape
    _, f, oh, ow = dout.shape
    kh, kw = w.shape[-2:]
    grad = dout.transpose(0, 2, 3, 1).reshape(-1, f)
    dw = (grad.T @ cols).reshape(w.shape)
    dcols = (grad @ w.reshape(f, -1)).reshape(n, oh, ow, c, kh, kw)
    dx = np.zeros((n, c, h+2*pad, width+2*pad), dtype=dcols.dtype)
    for i in range(kh):
        for j in range(kw):
            dx[:, :, i:i+oh*stride:stride, j:j+ow*stride:stride] += dcols[..., i, j].transpose(0, 3, 1, 2)
    return dx[:, :, pad:pad+h, pad:pad+width], dw, grad.sum(axis=0)


def max_pool_forward_fast(x, pool_param):
    ph, pw, stride = (pool_param[k] for k in ('pool_height', 'pool_width', 'stride'))
    windows = sliding_window_view(x, (ph, pw), axis=(2, 3))[:, :, ::stride, ::stride]
    flat = windows.reshape(*windows.shape[:4], ph*pw)
    arg = flat.argmax(axis=-1)
    return np.take_along_axis(flat, arg[..., None], axis=-1)[..., 0], (x.shape, arg, pw, stride)


def max_pool_backward_fast(dout, cache):
    shape, arg, pw, stride = cache
    dx = np.zeros(shape, dtype=dout.dtype)
    n, c, i, j = np.ogrid[tuple(slice(d) for d in dout.shape)]
    np.add.at(dx, (n, c, i*stride + arg//pw, j*stride + arg%pw), dout)
    return dx
