# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
Gradient-based learning applied to document recognition.
Proceedings of the IEEE (1998)
"""
import mxnet as mx
#from fit.ctc_loss import add_ctc_loss
#from fit.lstm import lstm
from ..config import config


from lstm import lstm
import simplenet
import fresnet as resnet
import fmobilenet as mobilenet




def crnn_no_lstm(network, per_batch_size):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    #net = eval(network).get_conv_feat() # b, c, h, w
    net, c = eval(network+".get_conv_feat")(data) # b, c, h, w

    # input
    #net = mx.sym.transpose(data=net, axes=[1,0,2,3])  # filter x bz x h x w
    #net = mx.sym.flatten(data=net) # filter x (bz x h x 25)
    #net = mx.sym.transpose(data=net, axes=[1,0]) # (bz x h x 25) x f

    #net = mx.sym.transpose(data=net, axes=[0,3,2,1])  # filter x bz x h x w
    #net = mx.sym.reshape(data=net, shape=(per_batch_size*config.seq_length, -1))  # (b*w, h*filter)

    net = mx.sym.transpose(data=net, axes=[3,2,0,1])  # w,h,batch_size,c
    #net = mx.sym.reshape(data=net, shape=(-3,-2)) # w, (h*batch_size), c
    net = mx.sym.reshape(data=net, shape=(-1,c)) # (w*h*batch_size), c


    # mx.sym.transpose(net, [])
    pred = mx.sym.FullyConnected(data=net, num_hidden=config.num_classes) # (bz x 25) x num_classes
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    return mx.sym.WarpCTC(data=pred, label=label, label_length=config.num_label, input_length=config.seq_length)


def crnn_lstm(network, per_batch_size):

    # input
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    net, _ = get_conv_feat(data) # b, c, h, w

    hidden_concat = lstm(net,num_lstm_layer=config.num_lstm_layer, num_hidden=config.num_hidden, seq_length=config.seq_length)

    # mx.sym.transpose(net, [])
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=config.num_classes) # (bz x 25) x num_classes

    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    return mx.sym.WarpCTC(data=pred, label=label, label_length=config.num_label, input_length=config.seq_length)



#from hyperparams.hyperparams import Hyperparams
#
#if __name__ == '__main__':
#    hp = Hyperparams()
#
#    init_states = {}
#    init_states['data'] = (hp.batch_size, 1, hp.img_height, hp.img_width)
#    init_states['label'] = (hp.batch_size, hp.num_label)
#
#    # init_c = {('l%d_init_c' % l): (hp.batch_size, hp.num_hidden) for l in range(hp.num_lstm_layer*2)}
#    # init_h = {('l%d_init_h' % l): (hp.batch_size, hp.num_hidden) for l in range(hp.num_lstm_layer*2)}
#    #
#    # for item in init_c:
#    #     init_states[item] = init_c[item]
#    # for item in init_h:
#    #     init_states[item] = init_h[item]
#
#    symbol = crnn_no_lstm(hp)
#    interals = symbol.get_internals()
#    _, out_shapes, _ = interals.infer_shape(**init_states)
#    shape_dict = dict(zip(interals.list_outputs(), out_shapes))
#
#    for item in shape_dict:
#        print(item,shape_dict[item])


