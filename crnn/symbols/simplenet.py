from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
import numpy as np
from ..config import config

def get_conv_feat(data):

    kernel_size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    padding_size = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
    layer_size = [min(32*2**(i+1), 512) for i in range(len(kernel_size))]
    def convRelu(i, input_data, bn=True):
        layer = mx.symbol.Convolution(name='conv-%d' % i, data=input_data, kernel=kernel_size[i], pad=padding_size[i],
                                      num_filter=layer_size[i])
        if bn:
            layer = mx.sym.BatchNorm(data=layer, name='batchnorm-%d' % i)
        layer = mx.sym.LeakyReLU(data=layer,name='leakyrelu-%d' % i)
        return layer

    net = convRelu(0, data) # bz x f x 32 x 200
    max = mx.sym.Pooling(data=net, name='pool-0_m', pool_type='max', kernel=(2, 2), stride=(2, 2))
    avg = mx.sym.Pooling(data=net, name='pool-0_a', pool_type='avg', kernel=(2, 2), stride=(2, 2))
    net = max - avg  # 16 x 100
    net = convRelu(1, net)
    net = mx.sym.Pooling(data=net, name='pool-1', pool_type='max', kernel=(2, 2), stride=(2, 2)) # bz x f x 8 x 50
    net = convRelu(2, net, True)
    net = convRelu(3, net)
    net = mx.sym.Pooling(data=net, name='pool-2', pool_type='max', kernel=(2, 2), stride=(2, 2)) # bz x f x 4 x 25
    net = convRelu(4, net, True)
    net = convRelu(5, net)
    c = 512
    if not config.no4x1pooling:
      net = mx.symbol.Pooling(data=net, kernel=(4, 1), pool_type='avg', name='pool1') # bz x f x 1 x 25
    else:
      c *= 4
    net = mx.symbol.Dropout(data=net, p=0.5)
    if config.no4x1pooling:
      net = mx.symbol.reshape(data=net, shape=(0,-3,1,-2))
    return net, c

