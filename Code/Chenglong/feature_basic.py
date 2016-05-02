# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: basic features

"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

import config
from utils import ngram_utils, nlp_utils, np_utils
from utils import time_utils, logging_utils, pkl_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


class DocId(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        obs_set = set(obs_corpus)
        self.encoder = dict(zip(obs_set, range(len(obs_set))))

    def __name__(self):
        return "DocId"

    def transform_one(self, obs, target, id):
        return self.encoder[obs]


class DocIdEcho(BaseEstimator):
    """For product_uid"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DocIdEcho"

    def transform_one(self, obs, target, id):
        return obs


class DocIdOneHot(BaseEstimator):
    """For linear model"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DocIdOneHot"

    def transform(self):
        lb = LabelBinarizer(sparse_output=True)
        return lb.fit_transform(self.obs_corpus)


"""
product_uid     int(obs > 164038 and obs <= 206650)
id              int(obs > 163700 and obs <= 221473)
In test, we have
#sample = 147406 for product_uid <= 206650
#sample = 19287 for product_uid
The majority will be in 1st and 2nd part.
In specific,
50K points of 147406 in public, and the rest 100K points in private.
"""
class ProductUidDummy1(BaseEstimator):
    """For product_uid"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "ProductUidDummy1"

    def transform_one(self, obs, target, id):
        return int(obs<163800)


class ProductUidDummy2(BaseEstimator):
    """For product_uid"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "ProductUidDummy2"

    def transform_one(self, obs, target, id):
        return int(obs>206650)


class ProductUidDummy3(BaseEstimator):
    """For product_uid"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "ProductUidDummy3"

    def transform_one(self, obs, target, id):
        return int(obs > 164038 and obs <= 206650)


class DocLen(BaseEstimator):
    """Length of document"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DocLen"

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        return len(obs_tokens)


class DocFreq(BaseEstimator):
    """Frequency of the document in the corpus"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.counter = Counter(obs_corpus)

    def __name__(self):
        return "DocFreq"

    def transform_one(self, obs, target, id):
        return self.counter[obs]


class DocEntropy(BaseEstimator):
    """Entropy of the document"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DocEntropy"

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        counter = Counter(obs_tokens)
        count = np.asarray(list(counter.values()))
        proba = count/np.sum(count)
        return np_utils._entropy(proba)


class DigitCount(BaseEstimator):
    """Count of digit in the document"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DigitCount"

    def transform_one(self, obs, target, id):
        return len(re.findall(r"\d", obs))


class DigitRatio(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DigitRatio"

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        return np_utils._try_divide(len(re.findall(r"\d", obs)), len(obs_tokens))


class UniqueCount_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "UniqueCount_%s"%self.ngram_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        return len(set(obs_ngrams))


class UniqueRatio_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "UniqueRatio_%s"%self.ngram_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        return np_utils._try_divide(len(set(obs_ngrams)), len(obs_ngrams))


#--------------------- Attribute based features ----------------------
class AttrCount(BaseEstimator):
    """obs_corpus is a list of list of attributes"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "AttrCount"

    def transform_one(self, obs, target, id):
        """obs is a list of attributes"""
        return len(obs)


class AttrBulletCount(BaseEstimator):
    """obs_corpus is a list of list of attributes"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "AttrBulletCount"

    def transform_one(self, obs, target, id):
        """obs is a list of attributes"""
        cnt = 0
        for lst in obs:
            if lst[0].startswith("bullet"):
                cnt += 1
        return cnt


class AttrBulletRatio(BaseEstimator):
    """obs_corpus is a list of list of attributes"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "AttrBulletRatio"

    def transform_one(self, obs, target, id):
        """obs is a list of attributes"""
        cnt = 0
        for lst in obs:
            if lst[0].startswith("bullet"):
                cnt += 1
        return np_utils._try_divide(cnt, len(obs))


class AttrNonBulletCount(BaseEstimator):
    """obs_corpus is a list of list of attributes"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "AttrNonBulletCount"

    def transform_one(self, obs, target, id):
        """obs is a list of attributes"""
        cnt = 0
        for lst in obs:
            if not lst[0].startswith("bullet"):
                cnt += 1
        return cnt


class AttrNonBulletRatio(BaseEstimator):
    """obs_corpus is a list of list of attributes"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "AttrNonBulletRatio"

    def transform_one(self, obs, target, id):
        """obs is a list of attributes"""
        cnt = 0
        for lst in obs:
            if not lst[0].startswith("bullet"):
                cnt += 1
        return np_utils._try_divide(cnt, len(obs))


class AttrHasProductHeight(BaseEstimator):
    """obs_corpus is a list of list of attributes"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "AttrHasProductHeight"

    def transform_one(self, obs, target, id):
        """obs is a list of attributes"""
        for lst in obs:
            if lst[0].find("product height") != -1:
                return 1
        return 0


class AttrHasProductWidth(BaseEstimator):
    """obs_corpus is a list of list of attributes"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "AttrHasProductWidth"

    def transform_one(self, obs, target, id):
        """obs is a list of attributes"""
        for lst in obs:
            if lst[0].find("product width") != -1:
                return 1
        return 0


class AttrHasProductLength(BaseEstimator):
    """obs_corpus is a list of list of attributes"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "AttrHasProductLength"

    def transform_one(self, obs, target, id):
        """obs is a list of attributes"""
        for lst in obs:
            if lst[0].find("product length") != -1:
                return 1
        return 0


class AttrHasProductDepth(BaseEstimator):
    """obs_corpus is a list of list of attributes"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "AttrHasProductDepth"

    def transform_one(self, obs, target, id):
        """obs is a list of attributes"""
        for lst in obs:
            if lst[0].find("product depth") != -1:
                return 1
        return 0


class AttrHasIndoorOutdoor(BaseEstimator):
    """obs_corpus is a list of list of attributes"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "AttrHasIndoorOutdoor"

    def transform_one(self, obs, target, id):
        """obs is a list of attributes"""
        for lst in obs:
            if lst[0].find("indoor outdoor") != -1:
                return 1
        return 0


#---------------- Main ---------------------------
def main():
    logname = "generate_feature_basic_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)

    ## basic
    generators = [DocId, DocLen, DocFreq, DocEntropy, DigitCount, DigitRatio]
    obs_fields = ["search_term", "product_title", "product_description", 
                "product_attribute", "product_brand", "product_color"]
    for generator in generators:
        param_list = []
        sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger)
        sf.go()

    ## for product_uid
    generators = [DocIdEcho, DocFreq, ProductUidDummy1, ProductUidDummy2, ProductUidDummy3]
    obs_fields = ["product_uid"]
    for generator in generators:
        param_list = []
        sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger)
        sf.go()

    ## unique count
    generators = [UniqueCount_Ngram, UniqueRatio_Ngram]
    obs_fields = ["search_term", "product_title", "product_description", 
    "product_attribute", "product_brand", "product_color"]
    ngrams = [1,2,3]
    for generator in generators:
        for ngram in ngrams:
            param_list = [ngram]
            sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger)
            sf.go()

    ## for product_attribute_list
    generators = [
        AttrCount, 
        AttrBulletCount, 
        AttrBulletRatio, 
        AttrNonBulletCount, 
        AttrNonBulletRatio,
        AttrHasProductHeight,
        AttrHasProductWidth,
        AttrHasProductLength,
        AttrHasProductDepth,
        AttrHasIndoorOutdoor,
    ]
    obs_fields = ["product_attribute_list"]
    for generator in generators:
        param_list = []
        sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger)
        sf.go()


if __name__ == "__main__":
    main()
