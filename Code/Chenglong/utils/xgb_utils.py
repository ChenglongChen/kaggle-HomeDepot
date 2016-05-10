# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for XGBoost models

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import xgboost as xgb


class XGBRegressor:
    def __init__(self, booster='gbtree', base_score=0., colsample_bylevel=1., 
                colsample_bytree=1., gamma=0., learning_rate=0.1, max_delta_step=0.,
                max_depth=6, min_child_weight=1., missing=None, n_estimators=100, 
                nthread=1, objective='reg:linear', reg_alpha=1., reg_lambda=0., 
                reg_lambda_bias=0., seed=0, silent=True, subsample=1.):
        self.param = {
            "objective": objective,
            "booster": booster,
            "eta": learning_rate,
            "max_depth": max_depth,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "alpha": reg_alpha,
            "lambda": reg_lambda,
            "lambda_bias": reg_lambda_bias,
            "seed": seed,
            "silent": 1 if silent else 0,
            "nthread": nthread,
            "max_delta_step": max_delta_step,
        }
        self.missing = missing if missing is not None else np.nan
        self.n_estimators = n_estimators
        self.base_score = base_score

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ("%s(booster=\'%s\', base_score=%f, colsample_bylevel=%f, \n"
                    "colsample_bytree=%f, gamma=%f, learning_rate=%f, max_delta_step=%f, \n"
                    "max_depth=%d, min_child_weight=%f, missing=\'%s\', n_estimators=%d, \n"
                    "nthread=%d, objective=\'%s\', reg_alpha=%f, reg_lambda=%f, \n"
                    "reg_lambda_bias=%f, seed=%d, silent=%d, subsample=%f)" % (
                    self.__class__.__name__,
                    self.param["booster"],
                    self.base_score,
                    self.param["colsample_bylevel"],
                    self.param["colsample_bytree"],
                    self.param["gamma"],
                    self.param["eta"],
                    self.param["max_delta_step"],
                    self.param["max_depth"],
                    self.param["min_child_weight"],
                    str(self.missing),
                    self.n_estimators,
                    self.param["nthread"],
                    self.param["objective"],
                    self.param["alpha"],
                    self.param["lambda"],
                    self.param["lambda_bias"],
                    self.param["seed"],
                    self.param["silent"],
                    self.param["subsample"],
                ))
        
    def fit(self, X, y, feature_names=None):
        data = xgb.DMatrix(X, label=y, missing=self.missing, feature_names=feature_names)
        data.set_base_margin(self.base_score*np.ones(X.shape[0]))
        self.model = xgb.train(self.param, data, self.n_estimators)
        return self

    def predict(self, X, feature_names=None):
        data = xgb.DMatrix(X, missing=self.missing, feature_names=feature_names)
        data.set_base_margin(self.base_score*np.ones(X.shape[0]))
        y_pred = self.model.predict(data)
        return y_pred

    def plot_importance(self):
        ax = xgb.plot_importance(self.model)
        self.save_topn_features()
        return ax

    def save_topn_features(self, fname="XGBRegressor_topn_features.txt", topn=-1):
        ax = xgb.plot_importance(self.model)
        yticklabels = ax.get_yticklabels()[::-1]
        if topn == -1:
            topn = len(yticklabels)
        else:
            topn = min(topn, len(yticklabels))
        with open(fname, "w") as f:
            for i in range(topn):
                f.write("%s\n"%yticklabels[i].get_text())


class XGBClassifier:
    def __init__(self, num_class=2, booster='gbtree', base_score=0., colsample_bylevel=1., 
                colsample_bytree=1., gamma=0., learning_rate=0.1, max_delta_step=0.,
                max_depth=6, min_child_weight=1., missing=None, n_estimators=100, 
                nthread=1, objective='multi:softprob', reg_alpha=1., reg_lambda=0., 
                reg_lambda_bias=0., seed=0, silent=True, subsample=1.):
        self.param = {
            "objective": objective,
            "booster": booster,
            "eta": learning_rate,
            "max_depth": max_depth,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "alpha": reg_alpha,
            "lambda": reg_lambda,
            "lambda_bias": reg_lambda_bias,
            "seed": seed,
            "silent": 1 if silent else 0,
            "nthread": nthread,
            "max_delta_step": max_delta_step,
            "num_class": num_class,
        }
        self.missing = missing if missing is not None else np.nan
        self.n_estimators = n_estimators
        self.base_score = base_score
        self.num_class = num_class

    def __str__(self):
        return self.__repr__()
        
    def __repr__(self):
        return ("%s(num_class=%d, booster=\'%s\', base_score=%f, colsample_bylevel=%f, \n"
                    "colsample_bytree=%f, gamma=%f, learning_rate=%f, max_delta_step=%f, \n"
                    "max_depth=%d, min_child_weight=%f, missing=\'%s\', n_estimators=%d, \n"
                    "nthread=%d, objective=\'%s\', reg_alpha=%f, reg_lambda=%f, \n"
                    "reg_lambda_bias=%f, seed=%d, silent=%d, subsample=%f)" % (
                    self.__class__.__name__,
                    self.num_class,
                    self.param["booster"],
                    self.base_score,
                    self.param["colsample_bylevel"],
                    self.param["colsample_bytree"],
                    self.param["gamma"],
                    self.param["eta"],
                    self.param["max_delta_step"],
                    self.param["max_depth"],
                    self.param["min_child_weight"],
                    str(self.missing),
                    self.n_estimators,
                    self.param["nthread"],
                    self.param["objective"],
                    self.param["alpha"],
                    self.param["lambda"],
                    self.param["lambda_bias"],
                    self.param["seed"],
                    self.param["silent"],
                    self.param["subsample"],
                ))

    def fit(self, X, y, feature_names=None):
        data = xgb.DMatrix(X, label=y, missing=self.missing, feature_names=feature_names)
        data.set_base_margin(self.base_score*np.ones(X.shape[0] * self.num_class))
        self.model = xgb.train(self.param, data, self.n_estimators)
        return self

    def predict_proba(self, X, feature_names=None):
        data = xgb.DMatrix(X, missing=self.missing, feature_names=feature_names)
        data.set_base_margin(self.base_score*np.ones(X.shape[0] * self.num_class))
        proba = self.model.predict(data)
        proba = proba.reshape(X.shape[0], self.num_class)
        return proba

    def predict(self, X, feature_names=None):
        proba = self.predict_proba(X, feature_names=feature_names)
        y_pred = np.argmax(proba, axis=1)
        return y_pred

    def plot_importance(self):
        ax = xgb.plot_importance(self.model)
        self.save_topn_features()
        return ax

    def save_topn_features(self, fname="XGBClassifier_topn_features.txt", topn=10):
        ax = xgb.plot_importance(self.model)
        yticklabels = ax.get_yticklabels()[::-1]
        if topn == -1:
            topn = len(yticklabels)
        else:
            topn = min(topn, len(yticklabels))
        with open(fname, "w") as f:
            for i in range(topn):
                f.write("%s\n"%yticklabels[i].get_text())


class HomedepotXGBClassifier(XGBClassifier):
    def __init__(self, booster='gbtree', base_score=0., colsample_bylevel=1., 
                colsample_bytree=1., gamma=0., learning_rate=0.1, max_delta_step=0.,
                max_depth=6, min_child_weight=1., missing=None, n_estimators=100, 
                nthread=1, objective='multi:softprob', reg_alpha=1., reg_lambda=0., 
                reg_lambda_bias=0., seed=0, silent=True, subsample=1.):
        super().__init__(num_class=1, booster=booster, base_score=base_score, 
                        colsample_bylevel=colsample_bylevel, colsample_bytree=colsample_bytree, 
                        gamma=gamma, learning_rate=learning_rate, max_delta_step=max_delta_step,
                        max_depth=max_depth, min_child_weight=min_child_weight, missing=missing, 
                        n_estimators=n_estimators, nthread=nthread, objective=objective, 
                        reg_alpha=reg_alpha, reg_lambda=reg_lambda, reg_lambda_bias=reg_lambda_bias, 
                        seed=seed, silent=silent, subsample=subsample)
        # encode relevance to label
        self.encoder = {
            1.00: 0,
            1.25: 1,
            1.33: 2,
            1.50: 3, 
            1.67: 4, 
            1.75: 5, 
            2.00: 6, 
            2.25: 7, 
            2.33: 8, 
            2.50: 9, 
            2.67: 10, 
            2.75: 11, 
            3.00: 12, 
        }
        # decode label to relevance
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.num_class = len(self.encoder.keys())
        self.param["num_class"] = self.num_class
        
    def fit(self, X, y):
        # encode relevance to label
        y = list(map(self.encoder.get, y))
        y = np.asarray(y, dtype=int)
        super().fit(X, y)
        return self

    def predict(self, X):
        y_pred = super().predict(X)
        # decode label to relevance
        y_pred = list(map(self.decoder.get, y_pred))
        y_pred = np.asarray(y_pred, dtype=float)
        return y_pred
