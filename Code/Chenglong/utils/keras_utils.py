# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for Keras models

"""

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Layer, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils


class KerasDNNRegressor:
    def __init__(self, input_dropout=0.2, hidden_layers=2, hidden_units=64, 
                hidden_activation="relu", hidden_dropout=0.5, batch_norm=None, 
                optimizer="adadelta", nb_epoch=10, batch_size=64):
        self.input_dropout = input_dropout
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.hidden_dropout = hidden_dropout
        self.batch_norm = batch_norm
        self.optimizer = optimizer
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.scaler = None
        self.model = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ("%s(input_dropout=%f, hidden_layers=%d, hidden_units=%d, \n"
                    "hidden_activation=\'%s\', hidden_dropout=%f, batch_norm=\'%s\', \n"
                    "optimizer=\'%s\', nb_epoch=%d, batch_size=%d)" % (
                    self.__class__.__name__,
                    self.input_dropout,
                    self.hidden_layers,
                    self.hidden_units,
                    self.hidden_activation,
                    self.hidden_dropout,
                    str(self.batch_norm),
                    self.optimizer,
                    self.nb_epoch,
                    self.batch_size,
                ))


    def fit(self, X, y):
        ## scaler
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        #### build model
        self.model = Sequential()
        ## input layer
        self.model.add(Dropout(self.input_dropout, input_shape=(X.shape[1],)))
        ## hidden layers
        first = True
        hidden_layers = self.hidden_layers
        while hidden_layers > 0:
            self.model.add(Dense(self.hidden_units))
            if self.batch_norm == "before_act":
                self.model.add(BatchNormalization())
            if self.hidden_activation == "prelu":
                self.model.add(PReLU())
            elif self.hidden_activation == "elu":
                self.model.add(ELU())
            else:
                self.model.add(Activation(self.hidden_activation))
            if self.batch_norm == "after_act":
                self.model.add(BatchNormalization())
            self.model.add(Dropout(self.hidden_dropout))
            hidden_layers -= 1

        ## output layer
        output_dim = 1
        output_act = "linear"
        self.model.add(Dense(output_dim))
        self.model.add(Activation(output_act))
        
        ## loss
        if self.optimizer == "sgd":
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(loss="mse", optimizer=sgd)
        else:
            self.model.compile(loss="mse", optimizer=self.optimizer)

        ## fit
        self.model.fit(X, y,
                    nb_epoch=self.nb_epoch, 
                    batch_size=self.batch_size,
                    validation_split=0, verbose=0)
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        y_pred = y_pred.flatten()
        return y_pred
