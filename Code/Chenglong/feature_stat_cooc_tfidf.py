# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: statistical cooccurrence (weighted) features
        - TF & normalized TF
        - TFIDF & normalized TFIDF
        - Okapi BM25

"""

import sys
import string
from collections import defaultdict

import numpy as np
import pandas as pd

import config
from utils import dist_utils, ngram_utils, nlp_utils, np_utils, pkl_utils
from utils import logging_utils, time_utils
from feature_base import BaseEstimator, PairwiseFeatureWrapper


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


# ----------------------------- TF ------------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocTF_Ngram(BaseEstimator):
    """Single aggregation features"""
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocTF_%s_%s"%(
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocTF_%s_%s"%(
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            val_list.append(s)
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list


# ----------------------------- Normalized TF ------------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocNormTF_Ngram(BaseEstimator):
    """Single aggregation features"""
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocNormTF_%s_%s"%(
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocNormTF_%s_%s"%(
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            val_list.append(np_utils._try_divide(s, len(target_ngrams)))
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list


# ------------------------------ TFIDF -----------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocTFIDF_Ngram(BaseEstimator):
    """Single aggregation features"""
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold
        self.df_dict = self._get_df_dict()
        
    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocTFIDF_%s_%s"%(
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocTFIDF_%s_%s"%(
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def _get_df_dict(self):
        # smoothing
        d = defaultdict(lambda : 1)
        for target in self.target_corpus:
            target_tokens = nlp_utils._tokenize(target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            for w in set(target_ngrams):
                d[w] += 1
        return d

    def _get_idf(self, word):
        return np.log((self.N - self.df_dict[word] + 0.5)/(self.df_dict[word] + 0.5))

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            val_list.append(s * self._get_idf(w1))
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list


# ------------------------------ Normalized TFIDF -----------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocNormTFIDF_Ngram(BaseEstimator):
    """Single aggregation features"""
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold
        self.df_dict = self._get_df_dict()
        
    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocNormTFIDF_%s_%s"%(
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocNormTFIDF_%s_%s"%(
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def _get_df_dict(self):
        # smoothing
        d = defaultdict(lambda : 1)
        for target in self.target_corpus:
            target_tokens = nlp_utils._tokenize(target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            for w in set(target_ngrams):
                d[w] += 1
        return d

    def _get_idf(self, word):
        return np.log((self.N - self.df_dict[word] + 0.5)/(self.df_dict[word] + 0.5))

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            val_list.append(np_utils._try_divide(s, len(target_ngrams)) * self._get_idf(w1))
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list


# ------------------------ BM25 ---------------------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocBM25_Ngram(BaseEstimator):
    """Single aggregation features"""
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="", 
                str_match_threshold=config.STR_MATCH_THRESHOLD, k1=config.BM25_K1, b=config.BM25_B):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.k1 = k1
        self.b = b
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold
        self.df_dict = self._get_df_dict()
        self.avg_ngram_doc_len = self._get_avg_ngram_doc_len()

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocBM25_%s_%s"%(
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocBM25_%s_%s"%(
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def _get_df_dict(self):
        # smoothing
        d = defaultdict(lambda : 1)
        for target in self.target_corpus:
            target_tokens = nlp_utils._tokenize(target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            for w in set(target_ngrams):
                d[w] += 1
        return d

    def _get_idf(self, word):
        return np.log((self.N - self.df_dict[word] + 0.5)/(self.df_dict[word] + 0.5))

    def _get_avg_ngram_doc_len(self):
        lst = []
        for target in self.target_corpus:
            target_tokens = nlp_utils._tokenize(target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            lst.append(len(target_ngrams))
        return np.mean(lst)

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        K = self.k1 * (1 - self.b + self.b * np_utils._try_divide(len(target_ngrams), self.avg_ngram_doc_len))
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            bm25 = s * self._get_idf(w1) * np_utils._try_divide(1 + self.k1, s + K)
            val_list.append(bm25)
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list


# ---------------------------- Main --------------------------------------
def main(which):
    logname = "generate_feature_stat_cooc_tfidf_%s_%s.log"%(which, time_utils._timestamp())
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)

    generators = []
    if which == "tf":
        generators.append( StatCoocTF_Ngram )
    elif which == "norm_tf":
        generators.append( StatCoocNormTF_Ngram )
    elif which == "tfidf":
        generators.append( StatCoocTFIDF_Ngram )
    elif which == "norm_tfidf":
        generators.append( StatCoocNormTFIDF_Ngram )
    elif which == "bm25":
        generators.append( StatCoocBM25_Ngram )


    obs_fields_list = []
    target_fields_list = []
    ## query in document
    obs_fields_list.append( ["search_term", "search_term_alt", "search_term_auto_corrected"][:1] )
    target_fields_list.append( ["product_title", "product_title_product_name", "product_description", "product_attribute", "product_brand", "product_color"] )
    ## document in query
    obs_fields_list.append( ["product_title", "product_title_product_name", "product_description", "product_attribute", "product_brand", "product_color"] )
    target_fields_list.append( ["search_term", "search_term_alt", "search_term_auto_corrected"][:1] )
    ngrams = [1,2,3,12,123][:3]
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        for generator in generators:
            for ngram in ngrams:
                param_list = [ngram, aggregation_mode]
                pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
                pf.go()


    obs_fields_list = []
    target_fields_list = []
    ## query in document
    obs_fields_list.append( ["search_term_product_name"] )
    target_fields_list.append( ["product_title_product_name"] )
    ngrams = [1,2]
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        for generator in generators:
            for ngram in ngrams:
                if ngram == 2:
                    # since product_name is of length 2, it makes no difference 
                    # for various aggregation as there is only one item
                    param_list = [ngram, "mean"]
                else:
                    param_list = [ngram, aggregation_mode]
                pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
                pf.go()


if __name__ == "__main__":
    main(sys.argv[1])
