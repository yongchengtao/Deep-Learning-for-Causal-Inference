#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：tarreg_mertics.py
@Author  ：yongchengtao
@Date    ：7/12/23 10:18 AM 
"""
import tensorflow as tf
from udf_metrics.aipw_metrics import AIPW_Metrics


class TarReg_Metrics(AIPW_Metrics):
    def __init__(self, data, verbose=0):
        super().__init__(data, verbose)

    def split_pred(self, concat_pred):
        preds = {}
        preds['y0_pred'] = self.data['y_scaler'].inverse_transform(concat_pred[:, 0].reshape(-1, 1))
        preds['y1_pred'] = self.data['y_scaler'].inverse_transform(concat_pred[:, 1].reshape(-1, 1))
        preds['t_pred'] = concat_pred[:, 2]
        preds['epsilon'] = concat_pred[:, 3]
        preds['phi'] = concat_pred[:, 4:]
        return preds

    def compute_hstar(self, y0_pred, y1_pred, t_pred, t_true, epsilons):
        # helper for calculating the targeted regularization cate
        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred
        cc = t_true / t_pred - (1 - t_true) / (1 - t_pred)
        h_star = y_pred + epsilons * cc
        return h_star

    def TARREG_CATE(self, concat_pred):
        # Final calculation of Targeted Regularization loss
        p = self.split_pred(concat_pred)
        t_pred = tf.math.sigmoid(p['t_pred'])
        t_pred = (t_pred + 0.001) / 1.002  # a little numerical stability trick implemented by Shi
        hstar_0 = self.compute_hstar(p['y0_pred'], p['y1_pred'], t_pred, tf.zeros_like(p['epsilon']), p['epsilon'])
        hstar_1 = self.compute_hstar(p['y0_pred'], p['y1_pred'], t_pred, tf.ones_like(p['epsilon']), p['epsilon'])
        return hstar_1 - hstar_0

    def on_epoch_end(self, epoch, logs={}):
        concat_pred = self.model.predict(self.data['x'])
        # Calculate Empirical Metrics
        aipw_pred = tf.reduce_mean(self.AIPW(concat_pred));
        tf.summary.scalar('aipw', data=aipw_pred, step=epoch)
        ate_pred = tf.reduce_mean(self.ATE(concat_pred));
        tf.summary.scalar('ate', data=ate_pred, step=epoch)
        tarreg_pred = tf.reduce_mean(self.TARREG_CATE(concat_pred));
        tf.summary.scalar('tarreg_pred', data=tarreg_pred, step=epoch)
        pehe_nn = self.PEHEnn(concat_pred);
        tf.summary.scalar('cate_nn_err', data=tf.sqrt(pehe_nn), step=epoch)

        # Simulation Metrics
        ate_true = tf.reduce_mean(self.data['mu_1'] - self.data['mu_0'])
        ate_err = tf.abs(ate_true - ate_pred);
        tf.summary.scalar('ate_err', data=ate_err, step=epoch)
        aipw_err = tf.abs(ate_true - aipw_pred);
        tf.summary.scalar('aipw_err', data=aipw_err, step=epoch)
        tarreg_err = tf.abs(ate_true - tarreg_pred);
        tf.summary.scalar('tarreg_err', data=tarreg_err, step=epoch)
        pehe = self.PEHE(concat_pred);
        tf.summary.scalar('cate_err', data=tf.sqrt(pehe), step=epoch)
        out_str = f' — ate_err: {ate_err:.4f}  — aipw_err: {aipw_err:.4f} — tarreg_err: {tarreg_err:.4f} — cate_err: {tf.sqrt(pehe):.4f} — cate_nn_err: {tf.sqrt(pehe_nn):.4f} '

        if self.verbose > 0: print(out_str)
