# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: wordnet similarity based features (veeerrry time consuming)
@note: in our final submission, we are only able to generate WordNet_Path_Similarity between
       search_term and product_title in reasonable time.
"""

"""
http://stackoverflow.com/questions/16877517/compare-similarity-of-terms-expressions-using-nltk
http://stackoverflow.com/questions/22031968/how-to-find-distance-between-two-synset-using-python-nltk-in-wordnet-hierarchy

#----------------------------------------------------------------------------------------
Path similarity, wup_similarity and lch_similarity, all of these should work 
since they are based on the distance between two synsets in the Wordnet hierarchy.

dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')

dog.path_similarity(cat)

dog.lch_similarity(cat)

dog.wup_similarity(cat)

#----------------------------------------------------------------------------------------
synset1.path_similarity(synset2):

Return a score denoting how similar two word senses are, based on the shortest 
path that connects the senses in the is-a (hypernym/hypnoym) taxonomy. The 
score is in the range 0 to 1, except in those cases where a path cannot be 
found (will only be true for verbs as there are many distinct verb taxonomies),
in which case -1 is returned. A score of 1 represents identity i.e. comparing
a sense with itself will return 1.

#----------------------------------------------------------------------------------------
synset1.lch_similarity(synset2), Leacock-Chodorow Similarity:

Return a score denoting how similar two word senses are, based on the shortest 
path that connects the senses (as above) and the maximum depth of the taxonomy 
in which the senses occur. The relationship is given as -log(p/2d) where p is 
the shortest path length and d the taxonomy depth.

#----------------------------------------------------------------------------------------
synset1.wup_similarity(synset2), Wu-Palmer Similarity:

Return a score denoting how similar two word senses are, based on the depth of the
two senses in the taxonomy and that of their Least Common Subsumer (most specific 
ancestor node). Note that at this time the scores given do not always agree with 
those given by Pedersen's Perl implementation of Wordnet Similarity.
"""

import string

import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn

import config
from utils import dist_utils, ngram_utils, nlp_utils, pkl_utils
from utils import logging_utils, time_utils
from feature_base import BaseEstimator, PairwiseFeatureWrapper


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


class WordNet_Similarity(BaseEstimator):
    """Double aggregation features"""
    def __init__(self, obs_corpus, target_corpus, metric="path", aggregation_mode_prev="", aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode, None, aggregation_mode_prev)
        self.metric = metric
        if self.metric == "path":
            self.metric_func = lambda syn1, syn2: wn.path_similarity(syn1, syn2)
        elif self.metric == "lch":
            self.metric_func = lambda syn1, syn2: wn.lch_similarity(syn1, syn2)
        elif self.metric == "wup":
            self.metric_func = lambda syn1, syn2: wn.wup_similarity(syn1, syn2)
        else:
            raise(ValueError("Wrong similarity metric: %s, should be one of path/lch/wup."%self.metric))
            
    def __name__(self):
        feat_name = []
        for m1 in self.aggregation_mode_prev:
            for m in self.aggregation_mode:
                n = "WordNet_%s_Similarity_%s_%s"%(
                    string.capwords(self.metric), string.capwords(m1), string.capwords(m))
                feat_name.append(n)
        return feat_name

    def _maximum_similarity_for_two_synset_list(self, syn_list1, syn_list2):
        s = 0.
        if syn_list1 and syn_list2:
            for syn1 in syn_list1:
                for syn2 in syn_list2:
                    try:
                        _s = self.metric_func(syn1, syn2)
                    except:
                        _s = config.MISSING_VALUE_NUMERIC
                    if _s and _s > s:
                        s = _s
        return s

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_synset_list = [wn.synsets(obs_token) for obs_token in obs_tokens]
        target_synset_list = [wn.synsets(target_token) for target_token in target_tokens]
        val_list = []
        for obs_synset in obs_synset_list:
            _val_list = []
            for target_synset in target_synset_list:
                _s = self._maximum_similarity_for_two_synset_list(obs_synset, target_synset)
                _val_list.append(_s)
            if len(_val_list) == 0:
                _val_list = [config.MISSING_VALUE_NUMERIC]
            val_list.append( _val_list )
        if len(val_list) == 0:
            val_list = [[config.MISSING_VALUE_NUMERIC]]
        return val_list


class WordNet_Path_Similarity(WordNet_Similarity):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode_prev="", aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, "path", aggregation_mode_prev, aggregation_mode)


class WordNet_Lch_Similarity(WordNet_Similarity):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode_prev="", aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, "lch", aggregation_mode_prev, aggregation_mode)


class WordNet_Wup_Similarity(WordNet_Similarity):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode_prev="", aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, "wup", aggregation_mode_prev, aggregation_mode)


# ---------------------------- Main --------------------------------------
def main():
    logname = "generate_feature_wordnet_similarity_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    #### NOTE: use data BEFORE STEMMING
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED)

    # WordNet_Lch_Similarity and WordNet_Wup_Similarity are not used in final submission
    generators = [
        WordNet_Path_Similarity,
        WordNet_Lch_Similarity,
        WordNet_Wup_Similarity,
    ][:1]
    obs_fields_list = []
    target_fields_list = []
    # only search_term and product_title are used in final submission
    obs_fields_list.append( ["search_term", "search_term_alt", "search_term_auto_corrected"][:1] )
    target_fields_list.append( ["product_title", "product_description", "product_attribute"][:1] )
    # double aggregation
    aggregation_mode_prev = ["mean", "max", "min", "median"]
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        for generator in generators:
            param_list = [aggregation_mode_prev, aggregation_mode]
            pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
            pf.go()


if __name__ == "__main__":
    main()
