# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: first and last ngram features
@note: in the final submission, we only used intersect count, NOT including intersect position.

"""

import re
import string

import numpy as np
import pandas as pd

import config
from utils import dist_utils, ngram_utils, nlp_utils, np_utils, pkl_utils
from utils import logging_utils, time_utils
from feature_base import BaseEstimator, PairwiseFeatureWrapper
from feature_intersect_position import _inter_pos_list, _inter_norm_pos_list


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


# -------------------------- Count ----------------------------------
class Count_Ngram_BaseEstimator(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, idx, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.idx = idx
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold

    def _get_match_count(self, obs, target, idx):
        cnt = 0
        if (len(obs) != 0) and (len(target) != 0):
            for word in target:
                if dist_utils._is_str_match(word, obs[idx], self.str_match_threshold):
                    cnt += 1
        return cnt

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        return self._get_match_count(obs_ngrams, target_ngrams, self.idx)


class FirstIntersectCount_Ngram(Count_Ngram_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, ngram, 0, aggregation_mode, str_match_threshold)
        
    def __name__(self):
        return "FirstIntersectCount_%s"%self.ngram_str


class LastIntersectCount_Ngram(Count_Ngram_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, ngram, -1, aggregation_mode, str_match_threshold)
        
    def __name__(self):
        return "LastIntersectCount_%s"%self.ngram_str


# ------------------------- Ratio -------------------------------------------
class Ratio_Ngram_BaseEstimator(Count_Ngram_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, idx, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, ngram, idx, aggregation_mode, str_match_threshold)
    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        return np_utils._try_divide(self._get_match_count(obs_ngrams, target_ngrams, self.idx), len(target_ngrams))


class FirstIntersectRatio_Ngram(Ratio_Ngram_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, ngram, 0, aggregation_mode, str_match_threshold)
        
    def __name__(self):
        return "FirstIntersectRatio_%s"%self.ngram_str


class LastIntersectRatio_Ngram(Ratio_Ngram_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, ngram, -1, aggregation_mode, str_match_threshold)
        
    def __name__(self):
        return "LastIntersectRatio_%s"%self.ngram_str


# -------------------- Position ---------------------
class Position_Ngram_BaseEstimator(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, idx, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.idx = idx
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        return _inter_pos_list(target_ngrams, [obs_ngrams[self.idx]])


class FirstIntersectPosition_Ngram(Position_Ngram_BaseEstimator):
    """Single aggregation features"""
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, ngram, 0, aggregation_mode)
        
    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "FirstIntersectPosition_%s_%s"%(
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["FirstIntersectPosition_%s_%s"%(
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name


class LastIntersectPosition_Ngram(Position_Ngram_BaseEstimator):
    """Single aggregation features"""
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, ngram, -1, aggregation_mode)
        
    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "LastIntersectPosition_%s_%s"%(
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["LastIntersectPosition_%s_%s"%(
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name


# -------------------------- Norm Position ----------------------------------
class NormPosition_Ngram_BaseEstimator(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, idx, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.idx = idx
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        return _inter_norm_pos_list(target_ngrams, [obs_ngrams[self.idx]])


class FirstIntersectNormPosition_Ngram(NormPosition_Ngram_BaseEstimator):
    """Single aggregation features"""
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, ngram, 0, aggregation_mode)
        
    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "FirstIntersectNormPosition_%s_%s"%(
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["FirstIntersectNormPosition_%s_%s"%(
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name


class LastIntersectNormPosition_Ngram(NormPosition_Ngram_BaseEstimator):
    """Single aggregation features"""
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, ngram, -1, aggregation_mode)
        
    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "LastIntersectNormPosition_%s_%s"%(
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["LastIntersectNormPosition_%s_%s"%(
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name


# ---------------------------- Main --------------------------------------
def run_count():
    logname = "generate_feature_first_last_ngram_count_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)

    generators = [
        FirstIntersectCount_Ngram, 
        LastIntersectCount_Ngram, 
        FirstIntersectRatio_Ngram, 
        LastIntersectRatio_Ngram, 
    ]

    obs_fields_list = []
    target_fields_list = []
    ## query in document
    obs_fields_list.append( ["search_term", "search_term_product_name", "search_term_alt", "search_term_auto_corrected"][:2] )
    target_fields_list.append( ["product_title", "product_title_product_name", "product_description", "product_attribute", "product_brand", "product_color"] )
    ## document in query
    obs_fields_list.append( ["product_title", "product_title_product_name", "product_description", "product_attribute", "product_brand", "product_color"] )
    target_fields_list.append( ["search_term", "search_term_product_name", "search_term_alt", "search_term_auto_corrected"][:2] )
    ngrams = [1,2,3,12,123][:3]
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        for generator in generators:
            for ngram in ngrams:
                param_list = [ngram]
                pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
                pf.go()


def run_position():
    logname = "generate_feature_first_last_ngram_position_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)

    generators = [
        FirstIntersectPosition_Ngram, 
        LastIntersectPosition_Ngram, 
        FirstIntersectNormPosition_Ngram, 
        LastIntersectNormPosition_Ngram, 
    ]

    obs_fields_list = []
    target_fields_list = []
    ## query in document
    obs_fields_list.append( ["search_term", "search_term_product_name", "search_term_alt", "search_term_auto_corrected"][:2] )
    target_fields_list.append( ["product_title", "product_title_product_name", "product_description", "product_attribute", "product_brand", "product_color"] )
    ## document in query
    obs_fields_list.append( ["product_title", "product_title_product_name", "product_description", "product_attribute", "product_brand", "product_color"] )
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
    run_count()
    # # not used in final submission
    # run_position()
