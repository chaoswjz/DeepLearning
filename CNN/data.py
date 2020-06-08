#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import datetime

class myDataset:
    def __init__(self, dir_path, batch_size):
        self._path = dir_path
        self._batch_size = batch_size
        self.ds_lst = tf.data.Dataset.list_files(self._path + "*").batch(self._batch_size)
        self._total = len(list(self.ds_lst))
        self.cls2num = {}
        self.train_set = None
        self.valid_set = None

    def splitData(self):
        train_num = int(0.8 * self._total)
        self.train_set = self.ds_lst.take(train_num).batch(self._batch_size)
        self.valid_set = self.ds_lst.skip(train_num).batch(self._batch_size)

    def getClass(self):
        for idx, cls in enumerate(os.listdir(self._path)):
            self.cls2num[cls] = idx
        return self.cls2num

def getLabel(filepath):
    parts = tf.strings.split(filepath, os.sep)
    label = tf.strings.split(parts[-1], '.') [0]
    target = 1 if label == "dog" else 0
    return tf.Variable(target, dtype=tf.int8)

def getImage(image):
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, 224, 224)
    return image

def loadData(filepath, train=True):
    if train:
        label = getLabel(filepath)
    image = tf.io.read_file(filepath)
    image = getImage(image)
    if train:
        return image, label
    else:
        return image
