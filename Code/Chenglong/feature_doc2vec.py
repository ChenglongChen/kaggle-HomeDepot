# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: doc2vec based features

"""

import gensim
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import config
from utils import dist_utils, ngram_utils, nlp_utils
from utils import logging_utils, time_utils, pkl_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper, PairwiseFeatureWrapper


class Doc2Vec_BaseEstimator(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, doc2vec_model, sent_label, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.model = doc2vec_model
        self.sent_label = sent_label
        self.model_prefix = model_prefix
        self.vector_size = doc2vec_model.vector_size

    def _get_vector(self, sent):
        try:
            vect = self.model.docvecs[self.sent_label[sent]]
        except:
            vect = np.zeros(self.vector_size, dtype=float)
        return vect

    def _get_cosine_sim(self, sent1, sent2):
        vect1 = self._get_vector(sent1)
        vect2 = self._get_vector(sent2)
        return dist_utils._cosine_sim(vect1, vect2)

    def _get_vdiff(self, sent1, sent2):
        vect1 = self._get_vector(sent1)
        vect2 = self._get_vector(sent2)
        return dist_utils._vdiff(vect1, vect2)

    def _get_rmse(self, sent1, sent2):
        vect1 = self._get_vector(sent1)
        vect2 = self._get_vector(sent2)
        return dist_utils._rmse(vect1, vect2)


class Doc2Vec_Vector(Doc2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, doc2vec_model, sent_label, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, doc2vec_model, sent_label, model_prefix, aggregation_mode)

    def __name__(self):
        return "Doc2Vec_%s_D%d_Vector"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_vector(obs)


class Doc2Vec_Vdiff(Doc2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, doc2vec_model, sent_label, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, doc2vec_model, sent_label, model_prefix, aggregation_mode)

    def __name__(self):
        return "Doc2Vec_%s_D%d_Vdiff"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_vdiff(obs, target)


class Doc2Vec_CosineSim(Doc2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, doc2vec_model, sent_label, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, doc2vec_model, sent_label, model_prefix, aggregation_mode)

    def __name__(self):
        return "Doc2Vec_%s_D%d_CosineSim"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_cosine_sim(obs, target)


class Doc2Vec_RMSE(Doc2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, doc2vec_model, sent_label, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, doc2vec_model, sent_label, model_prefix, aggregation_mode)

    def __name__(self):
        return "Doc2Vec_%s_D%d_RMSE"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_rmse(obs, target)


# -------------------------------- Main ----------------------------------
def main():
    logname = "generate_feature_doc2vec_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    #### NOTE: use data BEFORE STEMMING
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED)

    doc2vec_model_dirs = []
    model_prefixes = []
    ## doc2vec model trained with Homedepot dataset: brand/color/obs/title/description
    doc2vec_model_dirs.append( config.DOC2VEC_MODEL_DIR + "/Homedepot-doc2vec-D%d-min_count%d.model"%(config.EMBEDDING_DIM, config.EMBEDDING_MIN_COUNT) )
    model_prefixes.append( "Homedepot" )
    for doc2vec_model_dir, model_prefix in zip(doc2vec_model_dirs, model_prefixes):
        ## load model
        try:
            if ".bin" in doc2vec_model_dir:
                doc2vec_model = gensim.models.Doc2Vec.load_word2vec_format(doc2vec_model_dir, binary=True)
            if ".txt" in doc2vec_model_dir:
                doc2vec_model = gensim.models.Doc2Vec.load_word2vec_format(doc2vec_model_dir, binary=False)
            else:
                doc2vec_model = gensim.models.Doc2Vec.load(doc2vec_model_dir)
                doc2vec_model_sent_label = pkl_utils._load(doc2vec_model_dir+".sent_label")
        except:
            continue

        # ## standalone (not used in model building)
        # obs_fields = ["search_term", "search_term_alt", "product_title", "product_description", "product_attribute"]
        # generator = Doc2Vec_Vector
        # param_list = [doc2vec_model, doc2vec_model_sent_label, model_prefix]
        # sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger)
        # sf.go()

        ## pairwise
        generators = [
            Doc2Vec_CosineSim, 
            Doc2Vec_RMSE, 
            # Doc2Vec_Vdiff, 
        ]
        obs_fields_list = []
        target_fields_list = []
        obs_fields_list.append( ["search_term", "search_term_alt"][:1] )
        target_fields_list.append( ["product_title", "product_description", "product_attribute", "product_brand", "product_color"] )
        for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
            for generator in generators:
                param_list = [doc2vec_model, doc2vec_model_sent_label, model_prefix]
                pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
                pf.go()


if __name__ == "__main__":
    main()
