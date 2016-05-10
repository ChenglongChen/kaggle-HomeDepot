# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: convert .csv format dataframe features (from Igor&Kostia) to .pkl format features

"""

import os
import sys
import imp
from optparse import OptionParser

import scipy
import numpy as np
import pandas as pd

import config
from utils import pkl_utils


class TuringTestConverter:
    def __init__(self, fname, name):
        self.fname = fname
        self.name = name

    def convert(self):
        dfAll = pd.read_csv(self.fname)
        columns_to_drop = ["id", "product_uid", "relevance", "search_term", "product_title"]
        columns_to_drop = [col for col in columns_to_drop if col in dfAll.columns]
        dfAll.drop(columns_to_drop, axis=1, inplace=True)
        for col in dfAll.columns:
            pkl_utils._save("%s/TuringTest_%s_%s.pkl"%(config.FEAT_DIR, self.name, col), dfAll[col].values)


def main():
    d = {
        "df_basic_features.csv": "Basic",
        "df_brand_material_dummies.csv": "BrandMaterialDummy",
        "df_dist_new.csv": "Dist",
        "df_st_tfidf.csv": "StTFIDF",
        "df_tfidf_intersept_new.csv": "TFIDF",
        "df_thekey_dummies.csv": "TheKeyDummy",
        "df_word2vec_new.csv": "Word2Vec",
        "dld_features.csv": "DLD",
    }

    for k,v in d.items():
        converter = TuringTestConverter(
                        fname="%s/Turing_test/%s"%(config.FEAT_DIR, k),
                        name=v)
        converter.convert()


if __name__ == "__main__":
    main()
