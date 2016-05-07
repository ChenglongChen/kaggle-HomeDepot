# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: convert .csv format TSNE features to .pkl format

"""

import os

import pandas as pd

import config
from utils import pkl_utils


def main():
    fnames = [
        "TSNE_LSA100_Word_Unigram_Pair_search_term_x_product_title_100D",
        "TSNE_LSA100_Word_Bigram_Pair_search_term_x_product_title_100D",
        "TSNE_LSA100_Word_Obs_Unigram_Target_Unigram_Cooc_search_term_x_product_title_100D",
        "TSNE_LSA100_Word_Obs_Unigram_Target_Bigram_Cooc_search_term_x_product_title_100D",
    ]

    fnames = [os.path.join(config.FEAT_DIR, fname+".csv") for fname in fnames]

    for fname in fnames:
        df = pd.read_csv(fname, index=False)
        f = df.values
        pkl_utils._save(fname[:-4]+".pkl", f)


if __name__ == "__main__":
    main()
