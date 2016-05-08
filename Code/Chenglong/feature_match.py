# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: match based features

"""

import re
import string

import numpy as np
import pandas as pd

import config
from utils import dist_utils, ngram_utils, nlp_utils, np_utils
from utils import logging_utils, time_utils, pkl_utils
from feature_base import BaseEstimator, PairwiseFeatureWrapper


class MatchQueryCount(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        
    def __name__(self):
        return "MatchQueryCount"

    def _str_whole_word(self, str1, str2, i_):
        cnt = 0
        if len(str1) > 0 and len(str2) > 0:
            try:
                while i_ < len(str2):
                    i_ = str2.find(str1, i_)
                    if i_ == -1:
                        return cnt
                    else:
                        cnt += 1
                        i_ += len(str1)
            except:
                pass
        return cnt

    def transform_one(self, obs, target, id):
        return self._str_whole_word(obs, target, 0)


class MatchQueryRatio(MatchQueryCount):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        
    def __name__(self):
        return "MatchQueryRatio"

    def transform_one(self, obs, target, id):
        return np_utils._try_divide(super().transform_one(obs, target, id), len(target.split(" ")))


#------------- Longest match features -------------------------------
class LongestMatchSize(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        
    def __name__(self):
        return "LongestMatchSize"

    def transform_one(self, obs, target, id):
        return dist_utils._longest_match_size(obs, target)


class LongestMatchRatio(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        
    def __name__(self):
        return "LongestMatchRatio"

    def transform_one(self, obs, target, id):
        return dist_utils._longest_match_ratio(obs, target)


# --------------------------- Attribute based features -------------------------
class MatchAttrCount(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        
    def __name__(self):
        return "MatchAttrCount"

    def _str_whole_word(self, str1, str2, i_):
        cnt = 0
        if len(str1) > 0 and len(str2) > 0:
            try:
                while i_ < len(str2):
                    i_ = str2.find(str1, i_)
                    if i_ == -1:
                        return cnt
                    else:
                        cnt += 1
                        i_ += len(str1)
            except:
                pass
        return cnt

    def transform_one(self, obs, target, id):
        cnt = 0
        for o in obs.split(" "):
            for t in target:
                if not t[0].startswith("bullet"):
                    if self._str_whole_word(obs, t[0], 0):
                        cnt += 1
        return cnt


class MatchAttrRatio(MatchQueryCount):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        
    def __name__(self):
        return "MatchAttrRatio"

    def transform_one(self, obs, target, id):
        lo = len(obs.split(" "))
        lt = len([t[0] for t in target if not t[0].startswith("bullet")])
        return np_utils._try_divide(super().transform_one(obs, target, id), lo*lt)


class IsIndoorOutdoorMatch(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        
    def __name__(self):
        return "IsIndoorOutdoorMatch"

    def transform_one(self, obs, target, id):
        os = []
        if obs.find("indoor") != -1:
            os.append("indoor")
        if obs.find("outdoor") != -1:
            os.append("outdoor")

        cnt = 0
        for t in target:
            if t[0].find("indoor outdoor") != -1:
                cnt = 1
                ts = t[1].split(" ")
                for i in ts:
                    if i in os:
                        return 1
        if cnt == 0:
            return 0
        else:
            return -1


# ---------------------------- Main --------------------------------------
def main():
    logname = "generate_feature_match_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)
    
    generators = [
        MatchQueryCount, 
        MatchQueryRatio, 
        LongestMatchSize, 
        LongestMatchRatio, 
    ]
    obs_fields_list = []
    target_fields_list = []
    obs_fields_list.append( ["search_term", "search_term_product_name", "search_term_alt", "search_term_auto_corrected"][:2] )
    target_fields_list.append( ["product_title", "product_title_product_name", "product_description", "product_attribute", "product_brand", "product_color"] )
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        for generator in generators:
            param_list = []
            pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
            pf.go()

    # product_attribute_list
    generators = [
        MatchAttrCount, 
        MatchAttrRatio, 
        IsIndoorOutdoorMatch, 
    ]
    obs_fields_list = []
    target_fields_list = []
    obs_fields_list.append( ["search_term", "search_term_alt", "search_term_auto_corrected"][:1] )
    target_fields_list.append( ["product_attribute_list"] )
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        for generator in generators:
            param_list = []
            pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
            pf.go()


if __name__ == "__main__":
    main()
