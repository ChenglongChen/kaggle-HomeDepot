# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for numpy

"""

import sys

import numpy as np
from scipy.stats import pearsonr
from collections import Counter

sys.path.append("..")
import config


def _sigmoid(score):
    p = 1. / (1. + np.exp(-score))
    return p


def _logit(p):
    return np.log(p/(1.-p))


def _softmax(score):
    score = np.asarray(score, dtype=float)
    score = np.exp(score - np.max(score))
    score /= np.sum(score, axis=1)[:,np.newaxis]
    return score


def _cast_proba_predict(proba):
    N = proba.shape[1]
    w = np.arange(1,N+1)
    pred = proba * w[np.newaxis,:]
    pred = np.sum(pred, axis=1)
    return pred


def _one_hot_label(label, n_classes):
    num = label.shape[0]
    tmp = np.zeros((num, n_classes), dtype=int)
    tmp[np.arange(num),label.astype(int)] = 1
    return tmp


def _majority_voting(x, weight=None):
    ## apply weight
    if weight is not None:
        assert len(weight) == len(x)
        x = np.repeat(x, weight)
    c = Counter(x)
    value, count = c.most_common()[0]
    return value


def _voter(x, weight=None):
    idx = np.isfinite(x)
    if sum(idx) == 0:
        value = config.MISSING_VALUE_NUMERIC
    else:
        if weight is not None:
            value = _majority_voting(x[idx], weight[idx])
        else:
            value = _majority_voting(x[idx])
    return value


def _array_majority_voting(X, weight=None):
    y = np.apply_along_axis(_voter, axis=1, arr=X, weight=weight)
    return y


def _mean(x):
    idx = np.isfinite(x)
    if sum(idx) == 0:
        value = float(config.MISSING_VALUE_NUMERIC) # cast it to float to accommodate the np.mean
    else:
        value = np.mean(x[idx]) # this is float!
    return value


def _array_mean(X):
    y = np.apply_along_axis(_mean, axis=1, arr=X)
    return y


def _corr(x, y_train):
    if _dim(x) == 1:
        corr = pearsonr(x.flatten(), y_train)[0]
        if str(corr) == "nan":
            corr = 0.
    else:
        corr = 1.
    return corr


def _dim(x):
    d = 1 if len(x.shape) == 1 else x.shape[1]
    return d


def _entropy(proba):
    entropy = -np.sum(proba*np.log(proba))
    return entropy


def _try_divide(x, y, val=0.0):
    """try to divide two numbers"""
    if y != 0.0:
        val = float(x) / y
    return val
