# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: convert .pkl format LSA features to .csv format for using Rtsne package in R

"""

import os

import pandas as pd

import config
from utils import pkl_utils


def main():
    fnames = [
        "LSA100_Word_Unigram_Pair_search_term_x_product_title_100D",
        "LSA100_Word_Bigram_Pair_search_term_x_product_title_100D",
        "LSA100_Word_Obs_Unigram_Target_Unigram_Cooc_search_term_x_product_title_100D",
        "LSA100_Word_Obs_Unigram_Target_Bigram_Cooc_search_term_x_product_title_100D",
    ]

    fnames = [os.path.join(config.FEAT_DIR, fname+".pkl") for fname in fnames]

    for fname in fnames:
        f = pkl_utils._load(fname)
        columns = ["LSA%d"%(i+1) for i in range(f.shape[1])]
        pd.DataFrame(f, columns=columns).to_csv(fname[:-4]+".csv", index=False)


if __name__ == "__main__":
    main()
