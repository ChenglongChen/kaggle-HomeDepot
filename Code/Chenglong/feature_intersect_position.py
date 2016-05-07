# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: intersect position features

"""

import re
import string

import numpy as np
import pandas as pd

import config
from utils import dist_utils, ngram_utils, nlp_utils, np_utils
from utils import logging_utils, time_utils, pkl_utils
from feature_base import BaseEstimator, PairwiseFeatureWrapper


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


def _inter_pos_list(obs, target):
    """
        Get the list of positions of obs in target
    """
    pos_list = [0]
    if len(obs) != 0:
        pos_list = [i for i,o in enumerate(obs, start=1) if o in target]
        if len(pos_list) == 0:
            pos_list = [0]
    return pos_list


def _inter_norm_pos_list(obs, target):
    pos_list = _inter_pos_list(obs, target)
    N = len(obs)
    return [np_utils._try_divide(i, N) for i in pos_list]


class IntersectPosition_Ngram(BaseEstimator):
    """Single aggregation features"""
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        
    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "IntersectPosition_%s_%s"%(
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["IntersectPosition_%s_%s"%(
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        pos_list = _inter_pos_list(obs_ngrams, target_ngrams)
        return pos_list


class IntersectNormPosition_Ngram(BaseEstimator):
    """Single aggregation features"""
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        
    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "IntersectNormPosition_%s_%s"%(
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["IntersectNormPosition_%s_%s"%(
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        pos_list = _inter_norm_pos_list(obs_ngrams, target_ngrams)
        return pos_list


# ---------------------------- Main --------------------------------------
def main():
    logname = "generate_feature_intersect_position_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)

    generators = [
        IntersectPosition_Ngram, 
        IntersectNormPosition_Ngram, 
    ]
    obs_fields_list = []
    target_fields_list = []
    ## query in document
    obs_fields_list.append( ["search_term", "search_term_product_name", "search_term_alt", "search_term_auto_corrected"][:2] )
    target_fields_list.append( ["product_title", "product_title_product_name", "product_description", "product_attribute", "product_brand", "product_color"][1:2] )
    ## document in query
    obs_fields_list.append( ["product_title", "product_title_product_name", "product_description", "product_attribute", "product_brand", "product_color"][1:2] )
    target_fields_list.append( ["search_term", "search_term_product_name", "search_term_alt", "search_term_auto_corrected"][:2] )
    ngrams = [1,2,3,12,123][:3]
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        for generator in generators:
            for ngram in ngrams:
                param_list = [ngram, aggregation_mode]
                pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
                pf.go()


if __name__ == "__main__":
    main()
