# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: group based distance aggregated statistical features
@note: such features are not used in final submission

"""

import os
import string

import numpy as np

import config
from config import TRAIN_SIZE
from utils import dist_utils, ngram_utils, nlp_utils, np_utils
from utils import logging_utils, time_utils, pkl_utils
from feature_base import BaseEstimator


class GroupDistanceStat(BaseEstimator):
    """Single aggregation features"""
    def __init__(self, dist_list, group_id_list, dist_name, group_id_name, aggregation_mode=""):
        super().__init__(obs_corpus=dist_list, target_corpus=group_id_list, aggregation_mode=aggregation_mode)
        self.dist_name = dist_name
        self.group_id_name = group_id_name
        self.group_dist_dict = self._get_group_dist_dict()

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "GroupDistanceStat_%s_%s_%s"%(
                self.dist_name, self.group_id_name, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["GroupDistanceStat_%s_%s_%s"%(
                self.dist_name, self.group_id_name, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def _get_group_dist_dict(self):
        d = {}
        for dist, group_id in zip(self.obs_corpus, self.target_corpus):
            if not group_id in d:
                d[group_id] = []
            d[group_id].append(dist)
        return d

    def transform_one(self, obs, target, id):
        return self.group_dist_dict.get(target, [config.MISSING_VALUE_NUMERIC])


# -------------------------------- Main ----------------------------------
def main():
    logname = "generate_feature_group_distance_stat_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)
    y_train = dfAll["relevance"].values[:TRAIN_SIZE]

    group_id_names = ["DocId_search_term", "DocId_product_title", "DocIdEcho_product_uid"]

    match_list = [
    "MatchQueryCount",
    "MatchQueryRatio",
    "LongestMatchRatio",
    ]

    tfidf_list = [
    "StatCoocTF_Unigram_Mean", 
    "StatCoocTF_Unigram_Max",
    "StatCoocTF_Unigram_Min",
    # "StatCoocNormTF_Unigram_Mean", 
    # "StatCoocNormTF_Unigram_Max",
    # "StatCoocNormTF_Unigram_Min", 
    "StatCoocTFIDF_Unigram_Mean",
    "StatCoocTFIDF_Unigram_Max",
    "StatCoocTFIDF_Unigram_Min",
    "StatCoocBM25_Unigram_Mean",
    "StatCoocBM25_Unigram_Max",
    "StatCoocBM25_Unigram_Min",
    # "StatCoocTF_Bigram_Mean", 
    # "StatCoocTF_Bigram_Max",
    # "StatCoocTF_Bigram_Min",
    # "StatCoocNormTF_Bigram_Mean", 
    # "StatCoocNormTF_Bigram_Max",
    # "StatCoocNormTF_Bigram_Min",
    # "StatCoocTFIDF_Bigram_Mean",
    # "StatCoocTFIDF_Bigram_Max",
    # "StatCoocTFIDF_Bigram_Min",
    # "StatCoocBM25_Bigram_Mean",
    # "StatCoocBM25_Bigram_Max",
    # "StatCoocBM25_Bigram_Min",
    # "StatCoocTF_Trigram_Mean", 
    # "StatCoocTF_Trigram_Max",
    # "StatCoocTF_Trigram_Min",
    # "StatCoocNormTF_Trigram_Mean", 
    # "StatCoocNormTF_Trigram_Max",
    # "StatCoocNormTF_Trigram_Min", 
    # "StatCoocTFIDF_Trigram_Mean",
    # "StatCoocTFIDF_Trigram_Max",
    # "StatCoocTFIDF_Trigram_Min",
    # "StatCoocBM25_Trigram_Mean",
    # "StatCoocBM25_Trigram_Max",
    # "StatCoocBM25_Trigram_Min",
    ]
    intersect_ngram_count_list = [    
    "IntersectCount_Unigram", 
    "IntersectRatio_Unigram", 
    # "IntersectCount_Bigram", 
    # "IntersectRatio_Bigram", 
    # "IntersectCount_Trigram", 
    # "IntersectRatio_Trigram", 
    ]
    first_last_ngram_list = [
    "FirstIntersectCount_Unigram", 
    "FirstIntersectRatio_Unigram", 
    "LastIntersectCount_Unigram", 
    "LastIntersectRatio_Unigram",
    # "FirstIntersectCount_Bigram", 
    # "FirstIntersectRatio_Bigram", 
    # "LastIntersectCount_Bigram", 
    # "LastIntersectRatio_Bigram",
    # "FirstIntersectCount_Trigram", 
    # "FirstIntersectRatio_Trigram", 
    # "LastIntersectCount_Trigram", 
    # "LastIntersectRatio_Trigram",
    ]

    cooccurrence_ngram_count_list = [
    "CooccurrenceCount_Unigram", 
    "CooccurrenceRatio_Unigram", 
    # "CooccurrenceCount_Bigram", 
    # "CooccurrenceRatio_Bigram",
    # "CooccurrenceCount_Trigram", 
    # "CooccurrenceRatio_Trigram",
    ]

    ngram_jaccard_list = [
    "JaccardCoef_Unigram", 
    # "JaccardCoef_Bigram", 
    # "JaccardCoef_Trigram", 
    "DiceDistance_Unigram", 
    # "DiceDistance_Bigram", 
    # "DiceDistance_Trigram", 
    ]

    char_dist_sim_list = [
    "CharDistribution_CosineSim",
    "CharDistribution_KL",
    ]

    tfidf_word_ngram_cosinesim_list = [
    "TFIDF_Word_Unigram_CosineSim",
    # "TFIDF_Word_Bigram_CosineSim",
    # "TFIDF_Word_Trigram_CosineSim",
    ]
    tfidf_char_ngram_cosinesim_list = [
    # "TFIDF_Char_Bigram_CosineSim",
    # "TFIDF_Char_Trigram_CosineSim",
    "TFIDF_Char_Fourgram_CosineSim",
    # "TFIDF_Char_Fivegram_CosineSim",
    ]

    lsa_word_ngram_cosinesim_list = [
    "LSA100_Word_Unigram_CosineSim",
    # "LSA100_Word_Bigram_CosineSim",
    # "LSA100_Word_Trigram_CosineSim",
    ]
    lsa_char_ngram_cosinesim_list = [
    # "LSA100_Char_Bigram_CosineSim",
    # "LSA100_Char_Trigram_CosineSim",
    "LSA100_Char_Fourgram_CosineSim",
    # "LSA100_Char_Fivegram_CosineSim",
    ]

    doc2vec_list = [
    "Doc2Vec_Homedepot_D100_CosineSim",
    ]

    word2vec_list = [
    "Word2Vec_N_Similarity",
    "Word2Vec_Homedepot_D100_CosineSim_Mean_Mean",
    "Word2Vec_Homedepot_D100_CosineSim_Max_Mean",
    "Word2Vec_Homedepot_D100_CosineSim_Min_Mean",
    ]

    distance_generator_list = \
    match_list + \
    tfidf_list + \
    intersect_ngram_count_list + \
    first_last_ngram_list + \
    cooccurrence_ngram_count_list + \
    ngram_jaccard_list + \
    tfidf_word_ngram_cosinesim_list + \
    tfidf_char_ngram_cosinesim_list + \
    lsa_word_ngram_cosinesim_list + \
    lsa_char_ngram_cosinesim_list + \
    char_dist_sim_list + \
    word2vec_list + \
    doc2vec_list

    obs_fields_list = []
    target_fields_list = []
    ## query in document
    obs_fields_list.append( ["search_term"] )
    target_fields_list.append( ["product_title", "product_title_product_name"] )
    aggregation_mode = ["mean", "max", "min"]
    for group_id_name in group_id_names:
        group_id_list = pkl_utils._load(os.path.join(config.FEAT_DIR, group_id_name+"_1D.pkl"))
        for distance_generator in distance_generator_list:
            for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
                for obs_field in obs_fields:
                    for target_field in target_fields:
                        dist_name = "%s_%s_x_%s"%(distance_generator, obs_field, target_field)
                        try:
                            dist_list = pkl_utils._load(os.path.join(config.FEAT_DIR, dist_name+"_1D.pkl"))
                            ext = GroupDistanceStat(dist_list, group_id_list, dist_name, group_id_name, aggregation_mode)
                            x = ext.transform()
                            if isinstance(ext.__name__(), list):
                                for i,feat_name in enumerate(ext.__name__()):
                                    dim = 1
                                    fname = "%s_%dD"%(feat_name, dim)
                                    pkl_utils._save(os.path.join(config.FEAT_DIR, fname+config.FEAT_FILE_SUFFIX), x[:,i])
                                    corr = np_utils._corr(x[:TRAIN_SIZE,i], y_train)
                                    logger.info("%s (%dD): corr = %.6f"%(fname, dim, corr))
                        except:
                            logger.info("Skip %s"%dist_name)
                            pass


if __name__ == "__main__":
    main()
