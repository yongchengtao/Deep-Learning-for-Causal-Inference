#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：tar_reg_loss.py
@Author  ：yongchengtao
@Date    ：7/12/23 10:17 AM 
"""
import tensorflow as tf
from udf_loss.base_loss import Base_Loss


class TarReg_Loss(Base_Loss):
    # initialize instance attributes
    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.name = 'tarreg_loss'

    def split_pred(self, concat_pred):
        # generic helper to make sure we dont make mistakes
        preds = {}
        preds['y0_pred'] = concat_pred[:, 0]
        preds['y1_pred'] = concat_pred[:, 1]
        preds['t_pred'] = concat_pred[:, 2]
        preds['epsilon'] = concat_pred[:, 3]  # we're moving epsilon into slot three
        preds['phi'] = concat_pred[:, 4:]
        return preds

    def calc_hstar(self, concat_true, concat_pred):
        # step 2 above
        p = self.split_pred(concat_pred)
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        t_pred = tf.math.sigmoid(concat_pred[:, 2])
        t_pred = (t_pred + 0.001) / 1.002  # a little numerical stability trick implemented by Shi
        y_pred = t_true * p['y1_pred'] + (1 - t_true) * p['y0_pred']

        # calling it cc for "clever covariate" as in SuperLearner TMLE literature
        cc = t_true / t_pred - (1 - t_true) / (1 - t_pred)
        h_star = y_pred + p['epsilon'] * cc
        return h_star

    def call(self, concat_true, concat_pred):
        y_true = concat_true[:, 0]

        standard_loss = self.standard_loss(concat_true, concat_pred)
        h_star = self.calc_hstar(concat_true, concat_pred)
        # step 3 above
        targeted_regularization = tf.reduce_sum(tf.square(y_true - h_star))

        # final
        loss = standard_loss + self.beta * targeted_regularization
        return loss
