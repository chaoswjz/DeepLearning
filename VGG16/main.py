#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import myCNN
from data import myDataset, loadData
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import functools
import numpy as np

if __name__ == "__main__":
    lossobj = SparseCategoricalCrossentropy()
    optimizer = Adadelta(learning_rate=0.01)

    path = '/home/chaoswjz/Documents/PycharmProjects/CNN/'
    tds = myDataset(path + 'train/', 32)
    cls2num = {"cat": 0, "dog": 1}
    numcls = len(cls2num)
    train_ds = tds.ds_lst
    model = myCNN(numcls)
    model.compile(optimizer, lossobj, ['accuracy'])
    for i, batch in enumerate(train_ds):
        zipobj = map(loadData, batch)
        imgs, labels = list(zip(*zipobj))
        imgs = tf.convert_to_tensor(imgs)
        labels = tf.convert_to_tensor(labels)
        model.fit(imgs, labels)

    model.summary()

    testds = myDataset(path + 'test/', 32)
    test_ds = testds.ds_lst
    for i, batch in enumerate(train_ds):
        zipobj = map(functools.partial(loadData, train=False), batch)
        imgs = list(zipobj)
        imgs = tf.convert_to_tensor(imgs)

        preds = np.argmax(model.predict(imgs), axis=1)
        print(preds)
