# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: group relevance based distance features
@note: such features are not used in final submission

"""

import re
import string

import numpy as np
import pandas as pd

import config
from config import TRAIN_SIZE
from utils import dist_utils, ngram_utils, nlp_utils
from utils import logging_utils, pkl_utils, time_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper, PairwiseFeatureWrapper


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


# -------------------- Group by (obs, relevance) based distance features ----------------------------------- #
# Something related to Query Expansion
class GroupRelevance_Ngram_Jaccard(BaseEstimator):
    """Single aggregation features"""
    def __init__(self, obs_corpus, target_corpus, id_list, dfTrain, target_field, relevance, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode, id_list)
        self.dfTrain = dfTrain[dfTrain["relevance"] != 0].copy()
        self.target_field = target_field
        self.relevance = relevance
        self.relevance_str = self._relevance_to_str()
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "Group_%sRelevance_%s_Jaccard_%s"%(
                self.relevance_str, self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["Group_%sRelevance_%s_Jaccard_%s"%(
                self.relevance_str, self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def _relevance_to_str(self):
        if isinstance(self.relevance, float):
            return re.sub("\.", "d", str(self.relevance))
        else:
            return str(self.relevance)

    def transform_one(self, obs, target, id):
        df = self.dfTrain[self.dfTrain["search_term"] == obs].copy()
        val_list = [config.MISSING_VALUE_NUMERIC]
        if df is not None:
            df = df[df["id"] != id].copy()
            df = df[df["relevance"] == self.relevance].copy()
            if df is not None and df.shape[0] > 0:
                target_tokens = nlp_utils._tokenize(target, token_pattern)
                target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
                val_list = []
                for x in df[self.target_field]:
                    x_tokens = nlp_utils._tokenize(x, token_pattern)
                    x_ngrams = ngram_utils._ngrams(x_tokens, self.ngram)
                    val_list.append(dist_utils._jaccard_coef(x_ngrams, target_ngrams))
        return val_list


# -------------------------------- Main ----------------------------------
def main():
    logname = "generate_feature_group_distance_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)
    dfTrain = dfAll.iloc[:TRAIN_SIZE].copy()

    ## run python3 splitter.py first
    split = pkl_utils._load("%s/splits_level1.pkl"%config.SPLIT_DIR)
    n_iter = len(split)

    relevances_complete = [1, 1.25, 1.33, 1.5, 1.67, 1.75, 2, 2.25, 2.33, 2.5, 2.67, 2.75, 3]
    relevances = [1, 1.33, 1.67, 2, 2.33, 2.67, 3]
    ngrams = [1]
    obs_fields = ["search_term"]
    target_fields = ["product_title", "product_description"]
    aggregation_mode = ["mean", "std", "max", "min", "median"]

    ## for cv
    for i in range(n_iter):
        trainInd, validInd = split[i][0], split[i][1]
        dfTrain2 = dfTrain.iloc[trainInd].copy()
        sub_feature_dir = "%s/Run%d" % (config.FEAT_DIR, i+1)

        for target_field in target_fields:
            for relevance in relevances:
                for ngram in ngrams:
                    param_list = [dfAll["id"], dfTrain2, target_field, relevance, ngram, aggregation_mode]
                    pf = PairwiseFeatureWrapper(GroupRelevance_Ngram_Jaccard, dfAll, obs_fields, [target_field], param_list, sub_feature_dir, logger)
                    pf.go()

    ## for all
    sub_feature_dir = "%s/All" % (config.FEAT_DIR)
    for target_field in target_fields:
        for relevance in relevances:
            for ngram in ngrams:
                param_list = [dfAll["id"], dfTrain, target_field, relevance, ngram, aggregation_mode]
                pf = PairwiseFeatureWrapper(GroupRelevance_Ngram_Jaccard, dfAll, obs_fields, [target_field], param_list, sub_feature_dir, logger)
                pf.go()


if __name__ == "__main__":
    main()
