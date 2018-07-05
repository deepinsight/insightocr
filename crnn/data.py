from __future__ import print_function

import os
from PIL import Image
import cv2
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
import random
from config import config



class ImageIter(mx.io.DataIter):

    """
    Iterator class for generating captcha image data
    """
    def __init__(self, dataset_path, image_path, batch_size, shuffle, image_set, lstm_init_states = None):
        """
        Parameters
        ----------
        data_root: str
            root directory of images
        data_list: str
            a .txt file stores the image name and corresponding labels for each line
        batch_size: int
        name: str
        """
        super(ImageIter, self).__init__()
        self.batch_size = batch_size
        self.image_channel = 3
        if config.to_gray:
          self.image_channel = 1

        dataset_file = os.path.join(dataset_path, '%s.txt'%image_set)
        self.imglist = []
        for line in open(dataset_file, 'r'):
          img_lst = line.strip().split(' ')
          item = {}
          item['image_path'] = os.path.join(dataset_path, config.image_path, img_lst[0])
          item['label'] = np.zeros( (config.num_label,), dtype=np.int)
          for idx in range(1, len(img_lst)):
            labelid = int(img_lst[idx])
            assert labelid>0
            item['label'][idx-1] = labelid
          self.imglist.append(item)
        self.provide_label = [('label', (self.batch_size, config.num_label))]
        if config.use_lstm:
          self.init_states = lstm_init_states
          self.init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_states]
          print(self.init_states)
          self.provide_data = [('data', (batch_size, self.image_channel, config.img_height, config.img_width))] + self.init_states
        else:
          self.provide_data = [('data', (batch_size, self.image_channel, config.img_height, config.img_width))]
        self.resize_aug = mx.image.ForceResizeAug((config.img_width, config.img_height))
        self.cur = 0
        self.shuffle = shuffle
        self.seq = range(len(self.imglist))
        self.reset()


    def num_samples(self):
      return len(self.seq)

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.cur = 0
        if self.shuffle:
          random.shuffle(self.seq)

    def next_sample(self):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.cur >= len(self.seq):
            raise StopIteration
        idx = self.seq[self.cur]
        self.cur += 1
        return self.imglist[idx]

    def next(self):
        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        #self.nbatch+=1
        batch_size = self.batch_size
        #c, h, w = self.data_shape
        batch_data = nd.empty(self.provide_data[0][1])
        batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                item = self.next_sample()
                with open(item['image_path'], 'rb') as fin:
                    img = fin.read()
                try:
                    #if config.to_gray:
                    #  _data = mx.image.imdecode(img, flag=0) #to gray
                    #else:
                    #  _data = mx.image.imdecode(img)
                    #self.check_valid_image(_data)
                    img = np.fromstring(img, np.uint8)
                    if config.to_gray:
                      _data = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
                    else:
                      _data = cv2.imdecode(img, cv2.IMREAD_COLOR)
                      _data = cv2.cvtColor(_data, cv2.COLOR_BGR2RGB)
                    if _data.shape[0]!=config.img_height or _data.shape[1]!=config.img_width:
                      _data = cv2.resize(_data, (config.img_width, config.img_height) )
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                _data = mx.nd.array(_data)
                #print(_data.shape)
                #if _data.shape[0]!=config.img_height or _data.shape[1]!=config.img_width:
                #  _data = self.resize_aug(_data)
                #print(_data.shape)
                _data = _data.astype('float32')
                _data -= 127.5
                _data *= 0.0078125
                data = [_data]
                label = item['label']
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    #print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration

        data_all = [batch_data]
        if config.use_lstm:
          data_all += self.init_state_arrays
        return io.DataBatch(data_all, [batch_label], batch_size - i)

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data.shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))


class ImageIterLstm(mx.io.DataIter):

    """
    Iterator class for generating captcha image data
    """

    def __init__(self, data_root, data_list, batch_size, data_shape, num_label, lstm_init_states, name=None):
        """
        Parameters
        ----------
        data_root: str
            root directory of images
        data_list: str
            a .txt file stores the image name and corresponding labels for each line
        batch_size: int
        name: str
        """
        super(ImageIterLstm, self).__init__()
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_label = num_label

        self.init_states = lstm_init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in lstm_init_states]

        self.data_root = data_root
        self.dataset_lines = open(data_list).readlines()

        self.provide_data = [('data', (batch_size, 1, data_shape[1], data_shape[0]))] + lstm_init_states
        self.provide_label = [('label', (self.batch_size, self.num_label))]
        self.name = name

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
        data = []
        label = []
        cnt = 0
        for m_line in self.dataset_lines:
            img_lst = m_line.strip().split(' ')
            img_path = os.path.join(self.data_root, img_lst[0])

            cnt += 1
            img = Image.open(img_path).resize(self.data_shape, Image.BILINEAR).convert('L')
            img = np.array(img).reshape((1, self.data_shape[1], self.data_shape[0]))
            data.append(img)

            ret = np.zeros(self.num_label, int)
            for idx in range(1, len(img_lst)):
                ret[idx - 1] = int(img_lst[idx])

            label.append(ret)
            if cnt % self.batch_size == 0:
                data_all = [mx.nd.array(data)] + self.init_state_arrays
                label_all = [mx.nd.array(label)]
                data_names = ['data'] + init_state_names
                label_names = ['label']
                data = []
                label = []
                yield SimpleBatch(data_names, data_all, label_names, label_all)
                continue

    def reset(self):
        # if self.dataset_lst_file.seekable():
        #     self.dataset_lst_file.seek(0)
        random.shuffle(self.dataset_lines)

