from __future__ import print_function

import argparse
import logging
import os
import mxnet as mx

from crnn.data import ImageIter
from crnn.symbols.crnn import crnn_no_lstm, crnn_lstm
from crnn.fit.ctc_metrics import CtcMetrics
from crnn.config import config, default, generate_config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_args():
    # Parse command line arguments

    parser = argparse.ArgumentParser(description='Train CRNN network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--dataset-path', help='dataset path', default=default.dataset_path, type=str)
    # training
    parser.add_argument('--frequent', help='frequency of logging', default=default.frequent, type=int)
    parser.add_argument('--kvstore', help='the kv-store type', default=default.kvstore, type=str)
    parser.add_argument('--no-shuffle', help='disable random shuffle', action='store_true')
    parser.add_argument('--resume', help='continue training', action='store_true')
    # e2e
    parser.add_argument('--pretrained', help='pretrained model prefix', default=default.pretrained, type=str)
    parser.add_argument('--pretrained-epoch', help='pretrained model epoch', default=default.pretrained_epoch, type=int)
    parser.add_argument('--prefix', help='new model prefix', default=default.prefix, type=str)
    parser.add_argument('--begin-epoch', help='begin epoch of training, use with resume', default=0, type=int)
    parser.add_argument('--end-epoch', help='end epoch of training', default=default.epoch, type=int)
    parser.add_argument('--batch-size', help='batch-size', default=default.batch_size, type=int)
    parser.add_argument('--lr', help='base learning rate', default=default.lr, type=float)
    parser.add_argument('--lr-step', help='learning rate steps (in epoch)', default=default.lr_step, type=str)
    parser.add_argument('--no_ohem', help='disable online hard mining', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
      for i in xrange(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      print('use cpu')
    else:
      print('gpu num:', len(ctx))

    args.ctx_num = len(ctx)
    args.per_batch_size = args.batch_size//args.ctx_num
    #data_names = ['data'] + [x[0] for x in init_states]
    if config.use_lstm:
      init_c = [('l%d_init_c' % l, (args.batch_size, config.num_hidden)) for l in range(config.num_lstm_layer * 2)]
      init_h = [('l%d_init_h' % l, (args.batch_size, config.num_hidden)) for l in range(config.num_lstm_layer * 2)]
      init_states = init_c + init_h
      data_names = ['data'] + [x[0] for x in init_states]
      train_iter = ImageIter(dataset_path=args.dataset_path, image_path=config.image_path, batch_size=args.batch_size, shuffle = not args.no_shuffle, image_set='train', lstm_init_states=init_states)
      val_iter = ImageIter(dataset_path=args.dataset_path, image_path=config.image_path, batch_size = args.batch_size, shuffle = False, image_set='test', lstm_init_states=init_states)
      sym = crnn_lstm(args.network, args.per_batch_size)
    else:
      data_names = ['data']
      train_iter = ImageIter(dataset_path=args.dataset_path, image_path=config.image_path, batch_size=args.batch_size, shuffle = not args.no_shuffle, image_set='train')
      val_iter = ImageIter(dataset_path=args.dataset_path, image_path=config.image_path, batch_size = args.batch_size, shuffle = False, image_set='test')
      sym = crnn_no_lstm(args.network, args.per_batch_size)

    #head = '%(asctime)-15s %(message)s'
    #logging.basicConfig(level=logging.DEBUG, format=head)


    metrics = CtcMetrics(config.seq_length)

    arg_params = None
    aux_params = None

    module = mx.mod.Module(
            symbol = sym,
            data_names= data_names,
            label_names=['label'],
            context=ctx)

    if args.network[0]=='r' or args.network[0]=='y':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    elif args.network[0]=='i' or args.network[0]=='x':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) #inception
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    _rescale = 1.0/args.ctx_num
    base_lr = args.lr
    lr_factor = 0.5
    lr_epoch = [int(epoch) for epoch in args.lr_step.split(',')]
    lr_epoch_diff = [epoch - args.begin_epoch for epoch in lr_epoch if epoch > args.begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * train_iter.num_samples() / args.batch_size) for epoch in lr_epoch_diff]
    logger.info('lr %f lr_epoch_diff %s lr_iters %s' % (lr, lr_epoch_diff, lr_iters))
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    if config.use_lstm:
      optimizer = 'AdaDelta'
      optimizer_params = {'wd': 0.00001,
                          'learning_rate': base_lr,
                          'lr_scheduler': lr_scheduler,
                          'rescale_grad': (1.0 / args.ctx_num),
                          'clip_gradient': None}
    else:
      optimizer = 'sgd'
      optimizer_params = {'momentum': 0.9,
                          'wd': 0.0002,
                          'learning_rate': base_lr,
                          'lr_scheduler': lr_scheduler,
                          'rescale_grad': (1.0 / args.ctx_num),
                          'clip_gradient': None}
    module.fit(train_data=train_iter,
               eval_data=val_iter,
               begin_epoch=args.begin_epoch,
               num_epoch=args.end_epoch,
               # use metrics.accuracy or metrics.accuracy_lcs
               eval_metric=mx.metric.np(metrics.accuracy, allow_extra_outputs=True),
               optimizer=optimizer, optimizer_params=optimizer_params, 
               initializer=initializer,
               arg_params=arg_params,
               aux_params=aux_params,
               batch_end_callback=mx.callback.Speedometer(args.batch_size, args.frequent),
               epoch_end_callback=mx.callback.do_checkpoint(args.prefix),
               )


if __name__ == '__main__':
    main()

