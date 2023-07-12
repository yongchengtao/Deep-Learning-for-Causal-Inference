#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：aipw_metrics.py
@Author  ：yongchengtao
@Date    ：7/12/23 10:11 AM 
"""
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


def pdist2sq(x, y):
    x2 = tf.reduce_sum(x ** 2, axis=-1, keepdims=True)
    y2 = tf.reduce_sum(y ** 2, axis=-1, keepdims=True)
    dist = x2 + tf.transpose(y2, (1, 0)) - 2. * x @ tf.transpose(y, (1, 0))
    return dist


class AIPW_Metrics(Callback):
    def __init__(self, data, verbose=0):
        super(AIPW_Metrics, self).__init__()
        self.data = data  # feed the callback the full dataset
        self.verbose = verbose

        # needed for PEHEnn; Called in self.find_ynn
        self.data['o_idx'] = tf.range(self.data['t'].shape[0])
        self.data['c_idx'] = self.data['o_idx'][
            self.data['t'].squeeze() == 0]  # These are the indices of the control units
        self.data['t_idx'] = self.data['o_idx'][
            self.data['t'].squeeze() == 1]  # These are the indices of the treated units

    def split_pred(self, concat_pred):
        preds = {}
        preds['y0_pred'] = self.data['y_scaler'].inverse_transform(concat_pred[:, 0].reshape(-1, 1))
        preds['y1_pred'] = self.data['y_scaler'].inverse_transform(concat_pred[:, 1].reshape(-1, 1))
        preds['t_pred'] = concat_pred[:, 2]
        preds['phi'] = concat_pred[:, 3:]
        return preds

    def find_ynn(self, Phi):
        # helper for PEHEnn
        PhiC, PhiT = tf.dynamic_partition(Phi, tf.cast(tf.squeeze(self.data['t']), tf.int32),
                                          2)  # separate control and treated reps
        dists = tf.sqrt(pdist2sq(PhiC, PhiT))  # calculate squared distance then sqrt to get euclidean
        yT_nn_idx = tf.gather(self.data['c_idx'], tf.argmin(dists, axis=0),
                              1)  # get c_idxs of smallest distances for treated units
        yC_nn_idx = tf.gather(self.data['t_idx'], tf.argmin(dists, axis=1),
                              1)  # get t_idxs of smallest distances for control units
        yT_nn = tf.gather(self.data['y'], yT_nn_idx, 1)  # now use these to retrieve y values
        yC_nn = tf.gather(self.data['y'], yC_nn_idx, 1)
        y_nn = tf.dynamic_stitch([self.data['t_idx'], self.data['c_idx']], [yT_nn, yC_nn])  # stitch em back up!
        return y_nn

    def PEHEnn(self, concat_pred):
        p = self.split_pred(concat_pred)
        y_nn = self.find_ynn(p['phi'])  # now its 3 plus because
        cate_nn_err = tf.reduce_mean(
            tf.square((1 - 2 * self.data['t']) * (y_nn - self.data['y']) - (p['y1_pred'] - p['y0_pred'])))
        return cate_nn_err

    def ATE(self, concat_pred):
        p = self.split_pred(concat_pred)
        return p['y1_pred'] - p['y0_pred']

    def PEHE(self, concat_pred):
        # simulation only
        p = self.split_pred(concat_pred)
        cate_err = tf.reduce_mean(tf.square(((self.data['mu_1'] - self.data['mu_0']) - (p['y1_pred'] - p['y0_pred']))))
        return cate_err

    # THIS IS THE NEW PART
    def AIPW(self, concat_pred):
        p = self.split_pred(concat_pred)
        t_pred = tf.math.sigmoid(p['t_pred'])
        t_pred = (t_pred + 0.001) / 1.002  # a little numerical stability trick implemented by Shi
        y_pred = p['y0_pred'] * (1 - self.data['t']) + p['y1_pred'] * self.data['t']
        # cc stands for clever covariate which is I think what it's called in TMLE lit
        cc = self.data['t'] * (1.0 / p['t_pred']) - (1.0 - self.data['t']) / (1.0 - p['t_pred'])
        cate = cc * (self.data['y'] - y_pred) + p['y1_pred'] - p['y0_pred']
        return cate

    def on_epoch_end(self, epoch, logs={}):
        concat_pred = self.model.predict(self.data['x'])
        # Calculate Empirical Metrics
        ate_pred = tf.reduce_mean(self.ATE(concat_pred));
        tf.summary.scalar('ate', data=ate_pred, step=epoch)
        pehe_nn = self.PEHEnn(concat_pred);
        tf.summary.scalar('cate_nn_err', data=tf.sqrt(pehe_nn), step=epoch)
        aipw = tf.reduce_mean(self.AIPW(concat_pred));
        tf.summary.scalar('aipw', data=aipw, step=epoch)

        # Simulation Metrics
        ate_true = tf.reduce_mean(self.data['mu_1'] - self.data['mu_0'])
        ate_err = tf.abs(ate_true - ate_pred);
        tf.summary.scalar('ate_err', data=ate_err, step=epoch)
        pehe = self.PEHE(concat_pred);
        tf.summary.scalar('cate_err', data=tf.sqrt(pehe), step=epoch)
        aipw_err = tf.abs(ate_true - aipw);
        tf.summary.scalar('aipw_err', data=aipw_err, step=epoch)
        out_str = f' — ate_err: {ate_err:.4f}  — aipw_err: {aipw_err:.4f} — cate_err: {tf.sqrt(pehe):.4f} — cate_nn_err: {tf.sqrt(pehe_nn):.4f} '

        if self.verbose > 0: print(out_str)
