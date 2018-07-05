import numpy as np
from easydict import EasyDict as edict

config = edict()

config.num_label = 10
config.img_width = 280
config.img_height = 32
config.seq_length = config.img_width//8
config.image_path = 'images'
config.num_classes = 5990
config.num_hidden = 100
config.num_lstm_layer = 2
config.use_lstm = False
config.no4x1pooling=False
config.to_gray = True


#config.PIXEL_MEANS = np.array([103.939, 116.779, 123.68])


# network related params
#config.FIXED_PARAMS = ['conv1', 'conv2']

default = edict()
default.network = 'simplenet' # simplenet, resnet, mobilenet, densenet
default.dataset = 'chinese' #chinese or vgg
#default.network = 'resnet'
default.pretrained = ''
default.pretrained_epoch = 0
default.lr = 0.0001
default.lr_step = '5,10'
# default dataset
default.dataset_path = ''
# default training
default.frequent = 20
default.batch_size = 128
default.kvstore = 'device'
# default e2e
default.prefix = 'model/e2e'
default.epoch = 20

network = edict()

dataset = edict()

dataset.chinese = edict()
dataset.chinese.dataset_path = '/gpu/data2/jiaguo/ocr_cn_dataset'
dataset.chinese.prefix = 'model/chinese'

dataset.vgg = edict()
dataset.vgg.dataset_path = '/gpu/data2/jiaguo/ocr_dataset'
dataset.vgg.prefix = 'model/vgg'
dataset.vgg.num_label = 23
dataset.vgg.img_width = 96
dataset.vgg.img_height = 32
dataset.vgg.seq_length = dataset.vgg.img_width//8
dataset.vgg.num_classes = 62
dataset.vgg.no4x1pooling=False
dataset.vgg.to_gray = False


def generate_config(_network, _dataset):
    if _network in network:
      for k, v in network[_network].items():
          if k in config:
              config[k] = v
          elif k in default:
              default[k] = v
    if _dataset in dataset:
      for k, v in dataset[_dataset].items():
          if k in config:
              config[k] = v
          elif k in default:
              default[k] = v

