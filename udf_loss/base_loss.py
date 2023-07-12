#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：base_loss.py
@Author  ：yongchengtao
@Date    ：7/12/23 10:00 AM 
"""
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy


class Base_Loss(keras.losses.Loss):
    # initialize instance attributes
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.name = 'standard_loss'

    def split_pred(self, concat_pred):
        # generic helper to make sure we dont make mistakes
        preds = {}
        preds['y0_pred'] = concat_pred[:, 0]
        preds['y1_pred'] = concat_pred[:, 1]
        preds['t_pred'] = concat_pred[:, 2]
        preds['phi'] = concat_pred[:, 3:]
        return preds

    # for logging purposes only
    def treatment_acc(self, concat_true, concat_pred):
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        # Since this isn't used as a loss, I've used tf.reduce_mean for interpretability
        return tf.reduce_mean(binary_accuracy(t_true, tf.math.sigmoid(p['t_pred']), threshold=0.5))

    def treatment_bce(self, concat_true, concat_pred):
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        lossP = tf.reduce_sum(binary_crossentropy(t_true, p['t_pred'], from_logits=True))
        return lossP

    def regression_loss(self, concat_true, concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - p['y0_pred']))
        loss1 = tf.reduce_sum(t_true * tf.square(y_true - p['y1_pred']))
        return loss0 + loss1

    def standard_loss(self, concat_true, concat_pred):
        lossR = self.regression_loss(concat_true, concat_pred)
        lossP = self.treatment_bce(concat_true, concat_pred)
        return lossR + self.alpha * lossP

    # compute loss
    def call(self, concat_true, concat_pred):
        return self.standard_loss(concat_true, concat_pred)
