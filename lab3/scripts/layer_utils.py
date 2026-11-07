from .layers import affine_forward, affine_backward, relu_forward, relu_backward
from .fast_layers import (conv_forward_fast, conv_backward_fast,
                          max_pool_forward_fast, max_pool_backward_fast)


def affine_relu_forward(x, w, b):
    affine, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(affine)
    return out, (fc_cache, relu_cache)


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    return affine_backward(relu_backward(dout, relu_cache), fc_cache)


def conv_relu_forward(x, w, b, conv_param):
    conv, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(conv)
    return out, (conv_cache, relu_cache)


def conv_relu_backward(dout, cache):
    conv_cache, relu_cache = cache
    return conv_backward_fast(relu_backward(dout, relu_cache), conv_cache)


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    relu, conv_relu_cache = conv_relu_forward(x, w, b, conv_param)
    out, pool_cache = max_pool_forward_fast(relu, pool_param)
    return out, (conv_relu_cache, pool_cache)


def conv_relu_pool_backward(dout, cache):
    conv_relu_cache, pool_cache = cache
    return conv_relu_backward(max_pool_backward_fast(dout, pool_cache), conv_relu_cache)
