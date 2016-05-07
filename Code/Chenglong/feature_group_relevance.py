# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: group based relevance features
@note: such features are not used in final submission (except GroupRelevance_Size)

"""

import string

import numpy as np
import pandas as pd

import config
from config import TRAIN_SIZE
from utils import dist_utils, ngram_utils, nlp_utils, np_utils
from utils import logging_utils, time_utils, pkl_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper


class GroupRelevance(BaseEstimator):
    """Single aggregation features"""
    def __init__(self, obs_corpus, target_corpus, id_list, dfTrain, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode, id_list)
        self.dfTrain = dfTrain[dfTrain["relevance"] != 0].copy()

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "GroupRelevance_%s"%string.capwords(self.aggregation_mode)
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["GroupRelevance_%s"%string.capwords(m) for m in self.aggregation_mode]
        return feat_name

    def transform_one(self, obs, target, id):
        df = self.dfTrain[self.dfTrain["search_term"] == obs].copy()
        val_list = [config.MISSING_VALUE_NUMERIC]
        if df is not None:
            df = df[df["id"] != id].copy()
            if df is not None and df.shape[0] > 0:
                val_list = df["relevance"].values.tolist()
        return val_list


# -------------------------------- Main ----------------------------------
def main():
    logname = "generate_feature_group_relevance_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)
    dfTrain = dfAll.iloc[:TRAIN_SIZE].copy()

    ## run python3 splitter.py first
    split = pkl_utils._load("%s/splits_level1.pkl"%config.SPLIT_DIR)
    n_iter = len(split)

    ## for cv
    for i in range(n_iter):
        trainInd, validInd = split[i][0], split[i][1]
        dfTrain2 = dfTrain.iloc[trainInd].copy()
        sub_feature_dir = "%s/Run%d" % (config.FEAT_DIR, i+1)

        obs_fields = ["search_term", "product_title"][1:]
        aggregation_mode = ["mean", "std", "max", "min", "median", "size"]
        param_list = [dfAll["id"], dfTrain2, aggregation_mode]
        sf = StandaloneFeatureWrapper(GroupRelevance, dfAll, obs_fields, param_list, sub_feature_dir, logger)
        sf.go()

    ## for all
    sub_feature_dir = "%s/All" % (config.FEAT_DIR)
    obs_fields = ["search_term", "product_title"][1:]
    aggregation_mode = ["mean", "std", "max", "min", "median", "size"]
    param_list = [dfAll["id"], dfTrain, aggregation_mode]
    sf = StandaloneFeatureWrapper(GroupRelevance, dfAll, obs_fields, param_list, sub_feature_dir, logger)
    sf.go()


if __name__ == "__main__":
    main()
