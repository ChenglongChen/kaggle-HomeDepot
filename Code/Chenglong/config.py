# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: config for Homedepot project

"""

import os
import platform

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from utils import os_utils


# ---------------------- Overall -----------------------
TASK = "all"
# # for testing data processing and feature generation
# TASK = "sample"
SAMPLE_SIZE = 1000

# ------------------------ PATH ------------------------
ROOT_DIR = "../.."

DATA_DIR = "%s/Data"%ROOT_DIR
CLEAN_DATA_DIR = "%s/Clean"%DATA_DIR

FEAT_DIR = "%s/Feat"%ROOT_DIR
FEAT_FILE_SUFFIX = ".pkl"
FEAT_CONF_DIR = "./conf"

OUTPUT_DIR = "%s/Output"%ROOT_DIR
SUBM_DIR = "%s/Subm"%OUTPUT_DIR

LOG_DIR = "%s/Log"%ROOT_DIR
FIG_DIR = "%s/Fig"%ROOT_DIR
TMP_DIR = "%s/Tmp"%ROOT_DIR
THIRDPARTY_DIR = "%s/Thirdparty"%ROOT_DIR

# word2vec/doc2vec/glove
WORD2VEC_MODEL_DIR = "%s/word2vec"%DATA_DIR
GLOVE_WORD2VEC_MODEL_DIR = "%s/glove/gensim"%DATA_DIR
DOC2VEC_MODEL_DIR = "%s/doc2vec"%DATA_DIR

# index split
SPLIT_DIR = "%s/split"%DATA_DIR

# dictionary
WORD_REPLACER_DATA = "%s/dict/word_replacer.csv"%DATA_DIR

# colors
COLOR_DATA = "%s/dict/color_data.py"%DATA_DIR

# ------------------------ DATA ------------------------
# provided data
TRAIN_DATA = "%s/train.csv"%DATA_DIR
TEST_DATA = "%s/test.csv"%DATA_DIR
ATTR_DATA = "%s/attributes.csv"%DATA_DIR
DESC_DATA = "%s/product_descriptions.csv"%DATA_DIR
SAMPLE_DATA = "%s/sample_submission.csv"%DATA_DIR

ALL_DATA_RAW = "%s/all.raw.csv.pkl"%CLEAN_DATA_DIR
ALL_DATA_LEMMATIZED = "%s/all.lemmatized.csv.pkl"%CLEAN_DATA_DIR
ALL_DATA_LEMMATIZED_STEMMED = "%s/all.lemmatized.stemmed.csv.pkl"%CLEAN_DATA_DIR
INFO_DATA = "%s/info.csv.pkl"%CLEAN_DATA_DIR

# size
TRAIN_SIZE = 74067
if TASK == "sample":
    TRAIN_SIZE = SAMPLE_SIZE
TEST_SIZE = 166693
VALID_SIZE_MAX = 60000 # 0.7 * TRAIN_SIZE

TRAIN_MEAN = 2.381634
TRAIN_VAR = 0.285135

TEST_MEAN = TRAIN_MEAN
TEST_VAR = TRAIN_VAR

MEAN_STD_DICT = {
    1.00: 0.000, # Common: [1, 1, 1]
    1.25: 0.433, # Rare: [1,1,1,2]
    1.33: 0.471, # Common: [1, 1, 2]
    1.50: 0.866, # Rare: [1, 1, 1, 3]
    1.67: 0.471, # Common: [1, 2, 2]
    1.75: 0.829, # Rare: [1, 1, 2, 3]
    2.00: 0.000, # Common: [2, 2, 2], [1, 2, 3]
    2.25: 0.829, # Rare: [1,2,3,3]
    2.33: 0.471, # Common: [2, 2, 3]
    2.50: 0.500, # Rare: [2,2,3,3]
    2.67: 0.471, # Common: [2, 3, 3]
    2.75: 0.433, # Rare: [2,3,3,3]
    3.00: 0.000, # Common: [3, 3, 3]
}

# ------------------------ PARAM ------------------------
# attribute name and value SEPARATOR
ATTR_SEPARATOR = " | "

# cv
N_RUNS = 5
N_FOLDS = 1

# intersect count/match
STR_MATCH_THRESHOLD = 0.85

# correct query with google spelling check dict
# turn this on/off to have two versions of features/models
# which is useful for ensembling
GOOGLE_CORRECTING_QUERY = True

# auto correcting query (quite time consuming; not used in final submission)
AUTO_CORRECTING_QUERY = False

# query expansion (not used in final submission)
QUERY_EXPANSION = False

# bm25
BM25_K1 = 1.6
BM25_B = 0.75

# svd
SVD_DIM = 100
SVD_N_ITER = 5

# xgboost
# mean of relevance in training set
BASE_SCORE = TRAIN_MEAN

# word2vec/doc2vec
EMBEDDING_ALPHA = 0.025
EMBEDDING_LEARNING_RATE_DECAY = 0.5
EMBEDDING_N_EPOCH = 5
EMBEDDING_MIN_COUNT = 3
EMBEDDING_DIM = 100
EMBEDDING_WINDOW = 5
EMBEDDING_WORKERS = 6

# count transformer
COUNT_TRANSFORM = np.log1p

# missing value
MISSING_VALUE_STRING = "MISSINGVALUE"
MISSING_VALUE_NUMERIC = -1.

# stop words
STOP_WORDS = set(ENGLISH_STOP_WORDS)

# ------------------------ OTHER ------------------------
RANDOM_SEED = 2016
PLATFORM = platform.system()
NUM_CORES = 4 if PLATFORM == "Windows" else 14

DATA_PROCESSOR_N_JOBS = 4 if PLATFORM == "Windows" else 6
AUTO_SPELLING_CHECKER_N_JOBS = 4 if PLATFORM == "Windows" else 8
# multi processing is not faster
AUTO_SPELLING_CHECKER_N_JOBS = 1

## rgf
RGF_CALL_EXE = "%s/rgf1.2/test/call_exe.pl"%THIRDPARTY_DIR
RGF_EXTENSION = ".exe" if PLATFORM == "Windows" else ""
RGF_EXE = "%s/rgf1.2/bin/rgf%s"%(THIRDPARTY_DIR, RGF_EXTENSION)


# ---------------------- CREATE PATH --------------------
DIRS = []
DIRS += [CLEAN_DATA_DIR]
DIRS += [SPLIT_DIR]
DIRS += [FEAT_DIR, FEAT_CONF_DIR]
DIRS += ["%s/All"%FEAT_DIR]
DIRS += ["%s/Run%d"%(FEAT_DIR,i+1) for i in range(N_RUNS)]
DIRS += ["%s/Combine"%FEAT_DIR]
DIRS += [OUTPUT_DIR, SUBM_DIR]
DIRS += ["%s/All"%OUTPUT_DIR]
DIRS += ["%s/Run%d"%(OUTPUT_DIR,i+1) for i in range(N_RUNS)]
DIRS += [LOG_DIR, FIG_DIR, TMP_DIR]
DIRS += [WORD2VEC_MODEL_DIR, DOC2VEC_MODEL_DIR, GLOVE_WORD2VEC_MODEL_DIR]

os_utils._create_dirs(DIRS)
