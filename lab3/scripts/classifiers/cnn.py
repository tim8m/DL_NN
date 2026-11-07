import numpy as np
from ..layers import affine_forward, affine_backward, softmax_loss
from ..layer_utils import (conv_relu_pool_forward, conv_relu_pool_backward,
                           affine_relu_forward, affine_relu_backward)


class ThreeLayerConvNet:
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0,
                 dtype=np.float32):
        c, h, w = input_dim
        pad = (filter_size-1)//2
        self.conv_param = {'stride': 1, 'pad': pad}
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        pool_h, pool_w = (h+2*pad-filter_size+1)//2, (w+2*pad-filter_size+1)//2
        self.reg, self.dtype = reg, dtype
        shapes = [(num_filters, c, filter_size, filter_size),
                  (num_filters*pool_h*pool_w, hidden_dim), (hidden_dim, num_classes)]
        self.params = {}
        for i, shape in enumerate(shapes, 1):
            self.params[f'W{i}'] = (weight_scale*np.random.randn(*shape)).astype(dtype)
            self.params[f'b{i}'] = np.zeros(num_filters if i == 1 else shape[-1], dtype=dtype)

    def loss(self, X, y=None):
        p = self.params
        h1, c1 = conv_relu_pool_forward(X.astype(self.dtype, copy=False), p['W1'],
                                       p['b1'], self.conv_param, self.pool_param)
        h2, c2 = affine_relu_forward(h1, p['W2'], p['b2'])
        scores, c3 = affine_forward(h2, p['W3'], p['b3'])
        if y is None:
            return scores
        loss, dscores = softmax_loss(scores, y)
        dh2, dw3, db3 = affine_backward(dscores, c3)
        dh1, dw2, db2 = affine_relu_backward(dh2, c2)
        _, dw1, db1 = conv_relu_pool_backward(dh1, c1)
        grads = {}
        for i, (dw, db) in enumerate([(dw1, db1), (dw2, db2), (dw3, db3)], 1):
            w = p[f'W{i}']
            loss += 0.5*self.reg*np.sum(w*w)
            grads[f'W{i}'], grads[f'b{i}'] = dw + self.reg*w, db
        return loss, grads
