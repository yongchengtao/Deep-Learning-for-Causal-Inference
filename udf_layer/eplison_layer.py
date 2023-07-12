#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：eplison_layer.py
@Author  ：yongchengtao
@Date    ：7/12/23 10:15 AM 
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer


class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # note there is only one epsilon were just duplicating it for conformability
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]
