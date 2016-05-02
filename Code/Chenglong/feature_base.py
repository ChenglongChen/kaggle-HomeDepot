# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: base class for feature generation

"""

import os
import sys

import numpy as np

import config
from config import TRAIN_SIZE
from utils import np_utils, pkl_utils


# Since we have many features that measure the correlation/similarity/distance
# between search_term and product_title/product_description, we implement this base class.
class BaseEstimator:
    def __init__(self, obs_corpus, target_corpus, aggregation_mode, id_list=None, aggregation_mode_prev=""):
        self.obs_corpus = obs_corpus
        self.N = len(obs_corpus)
        # for standalone feature, we use the same interface, so better take care of it
        self.target_corpus = range(self.N) if target_corpus is None else target_corpus
        # id_list is used for group based relevance/distance features
        self.id_list = range(self.N) if id_list is None else id_list
        # aggregation for list features, e.g., intersect positions
        self.aggregation_mode, self.aggregator = self._check_aggregation_mode(aggregation_mode)
        self.aggregation_mode_prev, self.aggregator_prev = self._check_aggregation_mode(aggregation_mode_prev)
        self.double_aggregation = False
        if self.aggregator_prev != [None]:
            # the output of transform_one is a list of list, i.e., [[...], [...], [...]]
            # self.aggregator_prev is used to aggregate the inner list
            # This is used for the following features:
            # 1. EditDistance_Ngram
            # 2. CompressionDistance_Ngram
            # 3. Word2Vec_CosineSim
            # 4. WordNet_Path_Similarity, WordNet_Lch_Similarity, WordNet_Wup_Similarity
            # which are very time consuming to compute the inner list
            self.double_aggregation = True

    def _check_aggregation_mode(self, aggregation_mode):
        valid_aggregation_modes = ["", "size", "mean", "std", "max", "min", "median"]
        if isinstance(aggregation_mode, str):
            assert aggregation_mode.lower() in valid_aggregation_modes, "Wrong aggregation_mode: %s"%aggregation_mode
            aggregation_mode = [aggregation_mode.lower()]
        elif isinstance(aggregation_mode, list):
            for m in aggregation_mode:
                assert m.lower() in valid_aggregation_modes, "Wrong aggregation_mode: %s"%m
            aggregation_mode = [m.lower() for m in aggregation_mode]

        aggregator = [None if m == "" else getattr(np, m) for m in aggregation_mode]

        return aggregation_mode, aggregator

    def transform(self):
        # original score
        score = list(map(self.transform_one, self.obs_corpus, self.target_corpus, self.id_list))
        # aggregation
        if isinstance(score[0], list):
            if self.double_aggregation:
                # double aggregation
                res = np.zeros((self.N, len(self.aggregator_prev) * len(self.aggregator)), dtype=float)
                for m,aggregator_prev in enumerate(self.aggregator_prev):
                    for n,aggregator in enumerate(self.aggregator):
                        idx = m * len(self.aggregator) + n
                        for i in range(self.N):
                            # process in a safer way
                            try:
                                tmp = []
                                for l in score[i]:
                                    try:
                                        s = aggregator_prev(l)
                                    except:
                                        s = config.MISSING_VALUE_NUMERIC
                                    tmp.append(s)
                            except:
                                tmp = [ config.MISSING_VALUE_NUMERIC ]
                            try:
                                s = aggregator(tmp)
                            except:
                                s = config.MISSING_VALUE_NUMERIC
                            res[i,idx] = s
            else:
                # single aggregation
                res = np.zeros((self.N, len(self.aggregator)), dtype=float)
                for m,aggregator in enumerate(self.aggregator):
                    for i in range(self.N):
                        # process in a safer way
                        try:
                            s = aggregator(score[i])
                        except:
                            s = config.MISSING_VALUE_NUMERIC
                        res[i,m] = s
        else:
            res = np.asarray(score, dtype=float)
        return res


# Wrapper for generating standalone feature, e.g., 
# count of words in search_term
class StandaloneFeatureWrapper:
    def __init__(self, generator, dfAll, obs_fields, param_list, feat_dir, logger, force_corr=False):
        self.generator = generator
        self.dfAll = dfAll
        self.obs_fields = obs_fields
        self.param_list = param_list
        self.feat_dir = feat_dir
        self.logger = logger
        self.force_corr = force_corr

    def go(self):
        y_train = self.dfAll["relevance"].values[:TRAIN_SIZE]
        for obs_field in self.obs_fields:
            if obs_field not in self.dfAll.columns:
                self.logger.info("Skip %s"%obs_field)
                continue
            obs_corpus = self.dfAll[obs_field].values
            ext = self.generator(obs_corpus, None, *self.param_list)
            x = ext.transform()
            if isinstance(ext.__name__(), list):
                for i,feat_name in enumerate(ext.__name__()):
                    dim = 1
                    fname = "%s_%s_%dD"%(feat_name, obs_field, dim)
                    pkl_utils._save(os.path.join(self.feat_dir, fname+config.FEAT_FILE_SUFFIX), x[:,i])
                    corr = np_utils._corr(x[:TRAIN_SIZE,i], y_train)
                    self.logger.info("%s (%dD): corr = %.6f"%(fname, dim, corr))
            else:
                dim = np_utils._dim(x)
                fname = "%s_%s_%dD"%(ext.__name__(), obs_field, dim)
                pkl_utils._save(os.path.join(self.feat_dir, fname+config.FEAT_FILE_SUFFIX), x)
                if dim == 1:
                    corr = np_utils._corr(x[:TRAIN_SIZE], y_train)
                    self.logger.info("%s (%dD): corr = %.6f"%(fname, dim, corr))
                elif self.force_corr:
                    for j in range(dim):
                        corr = np_utils._corr(x[:TRAIN_SIZE,j], y_train)
                        self.logger.info("%s (%d/%dD): corr = %.6f"%(fname, j+1, dim, corr))


# Wrapper for generating pairwise feature, e.g., 
# intersect count of words between search_term and product_title
class PairwiseFeatureWrapper:
    def __init__(self, generator, dfAll, obs_fields, target_fields, param_list, feat_dir, logger, force_corr=False):
        self.generator = generator
        self.dfAll = dfAll
        self.obs_fields = obs_fields
        self.target_fields = target_fields
        self.param_list = param_list
        self.feat_dir = feat_dir
        self.logger = logger
        self.force_corr = force_corr

    def go(self):
        y_train = self.dfAll["relevance"].values[:TRAIN_SIZE]
        for obs_field in self.obs_fields:
            if obs_field not in self.dfAll.columns:
                self.logger.info("Skip %s"%obs_field)
                continue
            obs_corpus = self.dfAll[obs_field].values
            for target_field in self.target_fields:
                if target_field not in self.dfAll.columns:
                    self.logger.info("Skip %s"%target_field)
                    continue
                target_corpus = self.dfAll[target_field].values
                ext = self.generator(obs_corpus, target_corpus, *self.param_list)
                x = ext.transform()
                if isinstance(ext.__name__(), list):
                    for i,feat_name in enumerate(ext.__name__()):
                        dim = 1
                        fname = "%s_%s_x_%s_%dD"%(feat_name, obs_field, target_field, dim)
                        pkl_utils._save(os.path.join(self.feat_dir, fname+config.FEAT_FILE_SUFFIX), x[:,i])
                        corr = np_utils._corr(x[:TRAIN_SIZE,i], y_train)
                        self.logger.info("%s (%dD): corr = %.6f"%(fname, dim, corr))
                else:
                    dim = np_utils._dim(x)
                    fname = "%s_%s_x_%s_%dD"%(ext.__name__(), obs_field, target_field, dim)
                    pkl_utils._save(os.path.join(self.feat_dir, fname+config.FEAT_FILE_SUFFIX), x)
                    if dim == 1:
                        corr = np_utils._corr(x[:TRAIN_SIZE], y_train)
                        self.logger.info("%s (%dD): corr = %.6f"%(fname, dim, corr))
                    elif self.force_corr:
                        for j in range(dim):
                            corr = np_utils._corr(x[:TRAIN_SIZE,j], y_train)
                            self.logger.info("%s (%d/%dD): corr = %.6f"%(fname, j+1, dim, corr))
