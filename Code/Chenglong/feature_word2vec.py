# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: word2vec based features

"""

import re
import sys
import string

import gensim
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import config
from utils import dist_utils, ngram_utils, nlp_utils, np_utils, pkl_utils
from utils import logging_utils, time_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper, PairwiseFeatureWrapper


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


# ------------------------ Word2Vec Features -------------------------
class Word2Vec_BaseEstimator(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, 
        aggregation_mode="", aggregation_mode_prev=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode, None, aggregation_mode_prev)
        self.model = word2vec_model
        self.model_prefix = model_prefix
        self.vector_size = word2vec_model.vector_size

    def _get_valid_word_list(self, text):
        return [w for w in text.lower().split(" ") if w in self.model]

    def _get_importance(self, text1, text2):
        len_prev_1 = len(text1.split(" "))
        len_prev_2 = len(text2.split(" "))
        len1 = len(self._get_valid_word_list(text1))
        len2 = len(self._get_valid_word_list(text2))
        imp = np_utils._try_divide(len1+len2, len_prev_1+len_prev_2)
        return imp

    def _get_n_similarity(self, text1, text2):
        lst1 = self._get_valid_word_list(text1)
        lst2 = self._get_valid_word_list(text2)
        if len(lst1) > 0 and len(lst2) > 0:
            return self.model.n_similarity(lst1, lst2)
        else:
            return config.MISSING_VALUE_NUMERIC

    def _get_n_similarity_imp(self, text1, text2):
        sim = self._get_n_similarity(text1, text2)
        imp = self._get_importance(text1, text2)
        return sim * imp

    def _get_centroid_vector(self, text):
        lst = self._get_valid_word_list(text)
        centroid = np.zeros(self.vector_size)
        for w in lst:
            centroid += self.model[w]
        if len(lst) > 0:
            centroid /= float(len(lst))
        return centroid

    def _get_centroid_vdiff(self, text1, text2):
        centroid1 = self._get_centroid_vector(text1)
        centroid2 = self._get_centroid_vector(text2)
        return dist_utils._vdiff(centroid1, centroid2)

    def _get_centroid_rmse(self, text1, text2):
        centroid1 = self._get_centroid_vector(text1)
        centroid2 = self._get_centroid_vector(text2)
        return dist_utils._rmse(centroid1, centroid2)

    def _get_centroid_rmse_imp(self, text1, text2):
        rmse = self._get_centroid_rmse(text1, text2)
        imp = self._get_importance(text1, text2)
        return rmse * imp


class Word2Vec_Centroid_Vector(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_Centroid_Vector"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_centroid_vector(obs)


class Word2Vec_Importance(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_Importance"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_importance(obs, target)


class Word2Vec_N_Similarity(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_N_Similarity"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_n_similarity(obs, target)


class Word2Vec_N_Similarity_Imp(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_N_Similarity_Imp"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_n_similarity_imp(obs, target)


class Word2Vec_Centroid_RMSE(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)
        
    def __name__(self):
        return "Word2Vec_%s_D%d_Centroid_RMSE"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_centroid_rmse(obs, target)


class Word2Vec_Centroid_RMSE_IMP(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)
        
    def __name__(self):
        return "Word2Vec_%s_D%d_Centroid_RMSE_IMP"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_centroid_rmse_imp(obs, target)


class Word2Vec_Centroid_Vdiff(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)
        
    def __name__(self):
        return "Word2Vec_%s_D%d_Centroid_Vdiff"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_centroid_vdiff(obs, target)


class Word2Vec_CosineSim(Word2Vec_BaseEstimator):
    """Double aggregation features"""
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, 
        aggregation_mode="", aggregation_mode_prev=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, 
            aggregation_mode, aggregation_mode_prev)
        
    def __name__(self):
        feat_name = []
        for m1 in self.aggregation_mode_prev:
            for m in self.aggregation_mode:
                n = "Word2Vec_%s_D%d_CosineSim_%s_%s"%(
                    self.model_prefix, self.vector_size, string.capwords(m1), string.capwords(m))
                feat_name.append(n)
        return feat_name
    
    def transform_one(self, obs, target, id):
        val_list = []
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        for obs_token in obs_tokens:
            _val_list = []
            if obs_token in self.model:
                for target_token in target_tokens:
                    if target_token in self.model:
                        sim = dist_utils._cosine_sim(self.model[obs_token], self.model[target_token]) 
                        _val_list.append(sim)
            if len(_val_list) == 0:
                _val_list = [config.MISSING_VALUE_NUMERIC]
            val_list.append( _val_list )
        if len(val_list) == 0:
            val_list = [[config.MISSING_VALUE_NUMERIC]]
        return val_list


# ---------------------------- Main --------------------------------------
def main(which):
    logname = "generate_feature_word2vec_%s_%s.log"%(which, time_utils._timestamp())
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    #### NOTE: use data BEFORE STEMMinG
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED)

    word2vec_model_dirs = []
    model_prefixes = []
    if which == "homedepot":
        ## word2vec model trained with Homedepot dataset: brand/color/query/title/description
        word2vec_model_dirs.append( config.WORD2VEC_MODEL_DIR + "/Homedepot-word2vec-D%d-min_count%d.model"%(config.EMBEDDING_DIM, config.EMBEDDING_MIN_COUNT) )
        model_prefixes.append( "Homedepot" )
    elif which == "wikipedia":
        ## word2vec model pretrained with Wikipedia+Gigaword 5
        word2vec_model_dirs.append( config.GLOVE_WORD2VEC_MODEL_DIR + "/glove.6B.300d.txt" )
        model_prefixes.append( "Wikipedia" )
    elif which == "google":
        ## word2vec model pretrained with Google News
        word2vec_model_dirs.append( config.WORD2VEC_MODEL_DIR + "/GoogleNews-vectors-negative300.bin" )
        model_prefixes.append( "GoogleNews" )

    for word2vec_model_dir, model_prefix in zip(word2vec_model_dirs, model_prefixes):
        ## load model
        try:
            if ".bin" in word2vec_model_dir:
                word2vec_model = gensim.models.Word2Vec.load_word2vec_format(word2vec_model_dir, binary=True)
            elif ".txt" in word2vec_model_dir:
                word2vec_model = gensim.models.Word2Vec.load_word2vec_format(word2vec_model_dir, binary=False)
            else:
                word2vec_model = gensim.models.Word2Vec.load(word2vec_model_dir)
        except:
            continue

        # ## standalone (not used in model building)
        # obs_fields = ["search_term", "product_title", "product_description"]
        # generator = Word2Vec_Centroid_Vector
        # param_list = [word2vec_model, model_prefix]
        # sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger)
        # sf.go()

        ## pairwise
        generators = [
            Word2Vec_Importance,
            Word2Vec_N_Similarity, 
            Word2Vec_N_Similarity_Imp, 
            Word2Vec_Centroid_RMSE, 
            Word2Vec_Centroid_RMSE_IMP,
            # # not used in final submission
            # Word2Vec_Centroid_Vdiff, 
        ]
        obs_fields_list = []
        target_fields_list = []
        obs_fields_list.append( ["search_term", "search_term_alt", "search_term_auto_corrected"][:1] )
        target_fields_list.append( ["product_title", "product_description", "product_attribute", "product_brand", "product_color"] )
        for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
            for generator in generators:
                param_list = [word2vec_model, model_prefix]
                pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
                pf.go()

        ## cosine sim
        generators = [
            Word2Vec_CosineSim,
        ]
        # double aggregation
        aggregation_mode_prev = ["mean", "max", "min", "median"]
        aggregation_mode = ["mean", "std", "max", "min", "median"]
        for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
            for generator in generators:
                param_list = [word2vec_model, model_prefix, aggregation_mode, aggregation_mode_prev]
                pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
                pf.go()


if __name__ == "__main__":
    main(sys.argv[1])
