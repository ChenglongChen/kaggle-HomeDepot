# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for ensembler models

"""

import numpy as np


class Ensembler:
    def __init__(self, learner_list, weight_list):
        self.learner_list = learner_list
        self.weight_list = weight_list

    def __str__(self):
        return "Ensembler"

    def fit(self, X, y):
        for i in range(len(self.learner_list)):
            self.learner_list[i] = self.learner_list[i].fit(X, y)
        return self

    def predict(self, X):
        y_pred = np.zeros((X.shape[0]), dtype=float)
        for i in range(len(self.learner_list)):
            y_pred += self.weight_list[i] * self.learner_list[i].predict(X)
        y_pred /= float(sum(self.weight_list))
        return y_pred
