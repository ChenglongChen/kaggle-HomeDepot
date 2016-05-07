# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for EnsembleRegressor models

"""

import numpy as np


class EnsembleRegressor:
    def __init__(self, learner_dict):
        self.learner_dict = learner_dict

    def __str__(self):
        return "EnsembleRegressor"

    def fit(self, X, y):
        for learner_name in self.learner_dict.keys():
            l = self.learner_dict[learner_name]["learner"]
            self.learner_dict[learner_name]["learner"] = l.fit(X, y)
        return self

    def predict(self, X):
        y_pred = np.zeros((X.shape[0]), dtype=float)
        w_sum = 0.
        for learner_name in self.learner_dict.keys():
            l = self.learner_dict[learner_name]["learner"]
            w = self.learner_dict[learner_name]["weight"]
            y_pred += w * l.predict(X)
            w_sum += w
        y_pred /= w_sum
        return y_pred
