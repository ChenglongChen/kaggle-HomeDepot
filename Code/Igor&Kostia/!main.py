# -*- coding: utf-8 -*-
"""
The file that loads other files.

Theoretically, we can produce the necessary output by running this file only.
In fact we often needed to run files step-by-step and reboot computers sometimes.

Modelling files have to be run after the features are generated and saved.

Competition: HomeDepot Search Relevance
Author: Igor Buinyi
Team: Turing test
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from nltk.stem.snowball import SnowballStemmer
import nltk
from time import time
import re
import csv
import os
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
stoplist.append('till')
stoplist_wo_can=stoplist[:]
stoplist_wo_can.remove('can')


os.chdir("D:/HomeDepot/")

from homedepot_functions import *
from google_dict import *

t0 = time()
t1 = time()

if not os.path.exists("processing_text"):
    os.mkdir("processing_text")
if not os.path.exists("features"):
    os.mkdir("features")
if not os.path.exists("models"):
    os.mkdir("models")
if not os.path.exists("models_ensemble"):
    os.mkdir("models_ensemble")


### text preprocessing    
from text_processing import *
from text_processing_wo_google import *


### feature generation
# features from Igor
from feature_extraction1 import *
from feature_extraction1_wo_google import *

# features from Kostia
from grams_and_terms_features import * 
from dld_features import * 
from tfidf_by_st_features import * 
from word2vec import * 
from word2vec_without_google_dict import *


### Modelling from Igor
from generate_feature_importances import *
from generate_models import *
from generate_model_wo_google import *
from generate_ensemble_output_from_models import *
### End of modelling from Igor


#####################################
# Modelling from Kostia
"""
Ensemble generation was done by a different method.
This step is included in order to reproduce our submissions.
However, this such modelling might be considered as redundant 
as score improvement is small and such an improvement might be achived within the files
prepared by Igor if the proper feature lists are selected for the models.
"""
### modelling was done by the following script which selects random feature subsets
# from ensemble_script_random_version import *
### if the previous script is run, some filenames with model names have to be changed manually
### (The algorithm picks random numbers and then saves files with those number. 
### Then the proper files should be loaded for stacking.)

# the following line reproduces the output which is used in our submissions
from ensemble_script_imitation_version import *


### Selecting models for ensembling
from model_selecting import *
### End of modelling from Kostia
#########################################


"""
The outputs from Igor and Kostia were blended together
with weights 0.75*Igor + 0.25*Kostia
"""





