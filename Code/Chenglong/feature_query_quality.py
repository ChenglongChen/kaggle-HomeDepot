# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: query quality based features

"""

import re
import os
import string

import numpy as np
import pandas as pd

import config
from config import TRAIN_SIZE
from utils import dist_utils, ngram_utils, nlp_utils, np_utils
from utils import logging_utils, time_utils, pkl_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper
import google_spelling_checker_dict


class QueryQuality(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        
    def __name__(self):
        return "QueryQuality"

    def transform_one(self, obs, target, id):
        return dist_utils._edit_dist(obs, target)


class IsInGoogleDict(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        
    def __name__(self):
        return "IsInGoogleDict"

    def transform_one(self, obs, target, id):
        if obs in google_spelling_checker_dict.spelling_checker_dict:
            return 1.
        else:
            return 0.


# ---------------------------- Main --------------------------------------
def main():
    logname = "generate_feature_query_quality_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    obs_corpus = []
    query_suffix = []
    # raw
    dfAll = pkl_utils._load(config.ALL_DATA_RAW)
    obs_corpus.append(dfAll["search_term"].values)
    query_suffix.append("raw")
    # after processing    
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED)
    obs_corpus.append(dfAll["search_term"].values)
    query_suffix.append("lemmatized")
    # after extracting product_name in search_term
    obs_corpus.append(dfAll["search_term_product_name"].values)
    query_suffix.append("product_name")
    if "search_term_auto_corrected" in dfAll.columns:
        # after auto correction
        obs_corpus.append(dfAll["search_term_auto_corrected"].values)
        query_suffix.append("corrected")  
    # after stemming
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)
    obs_corpus.append(dfAll["search_term"].values)
    query_suffix.append("stemmed")

    y_train = dfAll["relevance"].values[:TRAIN_SIZE]
    for i in range(len(query_suffix)-1):
        for j in range(i+1, len(query_suffix)):
            ext = QueryQuality(obs_corpus[i], obs_corpus[j])
            x = ext.transform()
            dim = np_utils._dim(x)
            fname = "%s_%s_x_%s_%dD"%(ext.__name__(), query_suffix[i], query_suffix[j], dim)
            pkl_utils._save(os.path.join(config.FEAT_DIR, fname+config.FEAT_FILE_SUFFIX), x)
            corr = np_utils._corr(x[:TRAIN_SIZE], y_train)
            logger.info("%s (%dD): corr = %.6f"%(fname, dim, corr))

    # raw
    dfAll = pkl_utils._load(config.ALL_DATA_RAW)
    obs_fields = ["search_term"]
    param_list = []
    sf = StandaloneFeatureWrapper(IsInGoogleDict, dfAll, obs_fields, param_list, config.FEAT_DIR, logger)
    sf.go()


if __name__ == "__main__":
    main()
