#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Activation, Dropout, Layer

class myDense(Layer):
    def __init__(self, num_outputs):
        super(myDense, self).__init__()
        self._num_out = num_outputs

    def build(self, input_shape):
        self.w = self.add_weight("weights",
                shape=[int(input_shape[-1]), self._num_out])
        self.b = self.add_weight("bias",
                                 shape=[self._num_out])

    def call(self, x):
        return tf.add(tf.matmul(x, self.w), self.b)

class myActivation(Layer):
    def __init__(self, mode):
        super(myActivation, self).__init__()
        self._mode = mode

    def call(self, x):
        if self._mode == 'relu':
            return tf.nn.relu(x)
        elif self._mode == 'sigmoid':
            return tf.nn.sigmoid(x)
        elif self._mode == 'tanh':
            return tf.nn.tanh(x)
        elif self._mode == 'softmax':
            return tf.nn.softmax(x)
        else:
            return x

class myConv2d(Layer):
    def __init__(self, filters, kernel_size=[3, 3], stride=[1, 1], padding='same'):
        super(myConv2d, self).__init__()
        self._filters = filters

        self._kernel_size = None
        if isinstance(kernel_size, int):
            self._kernel_size = [kernel_size, kernel_size]
        else:
            self._kernel_size = kernel_size

        self._stride = None
        if isinstance(stride, int):
            self._stride = [1, stride, stride, 1]
        elif isinstance(stride, list) and len(stride) == 2:
            self._stride = [1, stride[0], stride[1], 1]
        else:
            self._stride = stride

        self._padding = padding.upper()

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[self._kernel_size[0], self._kernel_size[1], int(input_shape[-1]), self._filters])
        self.bias = self.add_weight("bias",
                                    shape=[self._filters])

    def call(self, x):
        return tf.add(tf.nn.conv2d(x, self.kernel, self._stride, self._padding), self.bias)

class myCNN(Model):
    def __init__(self, numcls):
        super(myCNN, self).__init__()
        self._numcls = numcls
        self.conv1 = myConv2d(32, 3, padding='same')
        self.activation1 = myActivation('relu')
        self.conv2 = myConv2d(32, 3, padding='same')
        self.activation2 = myActivation('relu')
        self.pool1 = MaxPool2D()
        self.dropout1 = Dropout(0.25)

        self.conv3 = myConv2d(64, 3, padding='same')
        self.activation3 = myActivation('relu')
        self.conv4 = myConv2d(64, 3, padding='same')
        self.activation4 = myActivation('relu')
        self.pool2 = MaxPool2D()
        self.dropout2 = Dropout(0.25)

        self.flat = Flatten()
        self.fc1 = myDense(512)
        self.activation5 = myActivation('relu')
        self.dropout3 = Dropout(0.25)
        self.fc2 = myDense(self._numcls)
        self.activation6 = myActivation('softmax')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)

        x = self.conv3(x)
        x = self.activation3(x)
        x = self.conv4(x)
        x = self.activation4(x)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)

        x = self.flat(x)
        x = self.fc1(x)
        x = self.activation5(x)
        x = self.dropout3(x, training=training)
        x = self.fc2(x)
        x = self.activation6(x)

        return x
