# -*- coding: utf-8 -*-
"""
This is config for HomeDepot Project: Igor&Kostia's part

Competition: HomeDepot Search Relevance
Author: Igor Buinyi
Team: Turing test
"""


import os
ROOT_DIR = os.getcwd()

DATA_DIR= "%s/data"%ROOT_DIR
PROCESSINGTEXT_DIR= "%s/processing_text"%ROOT_DIR
FEATURES_DIR= "%s/features"%ROOT_DIR
SAVEDMODELS_DIR= "%s/saved_models"%ROOT_DIR
MODELS_DIR= "%s/models"%ROOT_DIR
MODELSENSEMBLE_DIR= "%s/models_ensemble"%ROOT_DIR
FEATURESETS_DIR="%s/feature_sets"%ROOT_DIR


if not os.path.exists(PROCESSINGTEXT_DIR):
    os.mkdir(PROCESSINGTEXT_DIR)
if not os.path.exists(FEATURES_DIR):
    os.mkdir(FEATURES_DIR)
if not os.path.exists(SAVEDMODELS_DIR):
    os.mkdir(SAVEDMODELS_DIR)
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
if not os.path.exists(MODELSENSEMBLE_DIR):
    os.mkdir(MODELSENSEMBLE_DIR)   

