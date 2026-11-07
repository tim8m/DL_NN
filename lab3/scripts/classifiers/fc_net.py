import numpy as np
from ..layers import (affine_forward, affine_backward, relu_forward, relu_backward,
                      batchnorm_forward, batchnorm_backward, dropout_forward,
                      dropout_backward, softmax_loss)


class FullyConnectedNet:
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0, weight_scale=1e-2,
                 dtype=np.float32, seed=None):
        if normalization not in (None, 'batchnorm'):
            raise ValueError('Supported normalization: batchnorm or None')
        if not 0 < dropout <= 1:
            raise ValueError('Dropout keep probability must be in (0, 1]')
        self.normalization, self.reg, self.dtype = normalization, reg, dtype
        self.num_layers = len(hidden_dims) + 1
        self.use_dropout = dropout != 1
        self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed
        self.bn_params = [{'mode': 'train'} for _ in hidden_dims] if normalization else []
        self.params = {}
        dims = [input_dim, *hidden_dims, num_classes]
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            self.params[f'W{i}'] = (weight_scale * np.random.randn(d_in, d_out)).astype(dtype)
            self.params[f'b{i}'] = np.zeros(d_out, dtype=dtype)
            if normalization and i < self.num_layers:
                self.params[f'gamma{i}'] = np.ones(d_out, dtype=dtype)
                self.params[f'beta{i}'] = np.zeros(d_out, dtype=dtype)

    def loss(self, X, y=None):
        mode = 'test' if y is None else 'train'
        self.dropout_param['mode'] = mode
        for bn_param in self.bn_params:
            bn_param['mode'] = mode
        out = X.astype(self.dtype, copy=False)
        caches = []
        for i in range(1, self.num_layers + 1):
            out, affine = affine_forward(out, self.params[f'W{i}'], self.params[f'b{i}'])
            bn = relu = drop = None
            if i < self.num_layers:
                if self.normalization:
                    out, bn = batchnorm_forward(out, self.params[f'gamma{i}'],
                                                self.params[f'beta{i}'], self.bn_params[i-1])
                out, relu = relu_forward(out)
                if self.use_dropout:
                    out, drop = dropout_forward(out, self.dropout_param)
            caches.append((affine, bn, relu, drop))
        if y is None:
            return out

        loss, dout = softmax_loss(out, y)
        grads = {}
        for i in range(self.num_layers, 0, -1):
            affine, bn, relu, drop = caches[i-1]
            if drop is not None:
                dout = dropout_backward(dout, drop)
            if relu is not None:
                dout = relu_backward(dout, relu)
            if bn is not None:
                dout, grads[f'gamma{i}'], grads[f'beta{i}'] = batchnorm_backward(dout, bn)
            dout, dw, grads[f'b{i}'] = affine_backward(dout, affine)
            w = self.params[f'W{i}']
            loss += 0.5 * self.reg * np.sum(w*w)
            grads[f'W{i}'] = dw + self.reg*w
        return loss, grads


class TwoLayerNet(FullyConnectedNet):
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0):
        super().__init__([hidden_dim], input_dim, num_classes, reg=reg,
                         weight_scale=weight_scale, dtype=np.float64)
