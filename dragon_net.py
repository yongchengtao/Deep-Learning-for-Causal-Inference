#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：dragon_net.py
@Author  ：yongchengtao
@Date    ：7/11/23 10:55 PM 
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import Model

from sklearn.preprocessing import StandardScaler

import datetime

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.optimizers import SGD, Adam

from udf_loss.base_loss import Base_Loss
from udf_loss.tarreg_loss import TarReg_Loss
from udf_metrics.aipw_metrics import AIPW_Metrics
from udf_layer.eplison_layer import EpsilonLayer
from udf_metrics.tarreg_mertics import TarReg_Metrics


def make_aipw(input_dim, reg_l2):
    x = Input(shape=(input_dim,), name='input')
    # representation
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_1')(x)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_2')(phi)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_3')(phi)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='y0_hidden_1')(phi)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='y1_hidden_1')(phi)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='y0_hidden_2')(
        y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='y1_hidden_2')(
        y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    # propensity prediction
    # Note that the activation is actually sigmoid, but we will squish it in the loss function for numerical stability reasons
    t_prediction = Dense(units=1, activation=None, name='t_prediction')(phi)

    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_prediction, phi])
    model = Model(inputs=x, outputs=concat_pred)
    return model


def load_IHDP_data(training_data, testing_data, i=7):
    with open(training_data, 'rb') as trf, open(testing_data, 'rb') as tef:
        train_data = np.load(trf)
        test_data = np.load(tef)
        y = np.concatenate((train_data['yf'][:, i], test_data['yf'][:, i])).astype(
            'float32')  # most GPUs only compute 32-bit floats
        t = np.concatenate((train_data['t'][:, i], test_data['t'][:, i])).astype('float32')
        x = np.concatenate((train_data['x'][:, :, i], test_data['x'][:, :, i]), axis=0).astype('float32')
        mu_0 = np.concatenate((train_data['mu0'][:, i], test_data['mu0'][:, i])).astype('float32')
        mu_1 = np.concatenate((train_data['mu1'][:, i], test_data['mu1'][:, i])).astype('float32')

        data = {'x': x, 't': t, 'y': y, 't': t, 'mu_0': mu_0, 'mu_1': mu_1}
        data['t'] = data['t'].reshape(-1, 1)  # we're just padding one dimensional vectors with an additional dimension
        data['y'] = data['y'].reshape(-1, 1)

        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data['y_scaler'] = StandardScaler().fit(data['y'])
        data['ys'] = data['y_scaler'].transform(data['y'])

    return data


def run_aipw(data):
    val_split = 0.2
    batch_size = 64
    verbose = 1
    i = 0
    tf.random.set_seed(i)
    np.random.seed(i)
    yt = np.concatenate([data['ys'], data['t']], 1)

    # Clear any logs from previous runs

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    sgd_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0),
        tensorboard_callback,
        AIPW_Metrics(data, verbose=verbose)
    ]

    sgd_lr = 1e-5
    momentum = 0.9

    aipw_model = make_aipw(data['x'].shape[1], .01)
    aipw_loss = Base_Loss(alpha=1.0)

    aipw_model.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True),
                       loss=aipw_loss,
                       metrics=[aipw_loss, aipw_loss.regression_loss, aipw_loss.treatment_acc]
                       )

    aipw_model.fit(x=data['x'], y=yt,
                   callbacks=sgd_callbacks,
                   validation_split=val_split,
                   epochs=300,
                   batch_size=batch_size,
                   verbose=verbose)


def make_dragonnet(input_dim, reg_l2):
    x = Input(shape=(input_dim,), name='input')
    # representation
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_1')(x)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_2')(phi)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_3')(phi)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='y0_hidden_1')(phi)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='y1_hidden_1')(phi)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='y0_hidden_2')(
        y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='y1_hidden_2')(
        y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    # propensity prediction
    # Note that the activation is actually sigmoid, but we will squish it in the loss function for numerical stability reasons
    t_predictions = Dense(units=1, activation=None, name='t_prediction')(phi)
    # Although the epsilon layer takes an input, it really just houses a free parameter.
    epsilons = EpsilonLayer()(t_predictions)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons, phi])
    model = Model(inputs=x, outputs=concat_pred)
    return model


def run_dragon_net(data):
    val_split = 0.2
    batch_size = 64
    verbose = 1
    i = 0
    tf.random.set_seed(i)
    np.random.seed(i)
    yt = np.concatenate([data['ys'], data['t']], 1)

    # Clear any logs from previous runs
    # !rm - rf. / logs /
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

    sgd_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0),
        tensorboard_callback,
        TarReg_Metrics(data, verbose=verbose)]

    sgd_lr = 1e-5
    momentum = 0.9

    dragonnet_model = make_dragonnet(data['x'].shape[1], .01)
    tarreg_loss = TarReg_Loss(alpha=1)

    dragonnet_model.compile(optimizer=SGD(learning_rate=sgd_lr, momentum=momentum, nesterov=True),
                            loss=tarreg_loss,
                            metrics=[tarreg_loss, tarreg_loss.regression_loss, tarreg_loss.treatment_acc])

    dragonnet_model.fit(x=data['x'], y=yt,
                        callbacks=sgd_callbacks,
                        validation_split=val_split,
                        epochs=300,
                        batch_size=batch_size,
                        verbose=verbose)


def main():
    data = load_IHDP_data(training_data='./ihdp_npci_1-100.train.npz', testing_data='./ihdp_npci_1-100.test.npz')

    run_aipw(data)

    # run_dragon_net(data)


if __name__ == '__main__':
    main()
