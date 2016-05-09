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

import os
os.chdir("g:/kaggle/latex/py")


from homedepot_functions import *
from google_dict import *


### text preprocessing    
from text_processing import *


### feature generation
# features from Igor
from feature_extraction1 import *

# features from Kostia
from grams_and_terms_features import * 
from dld_features import * 
from tfidf_by_st_features import * 
from word2vec import * 

"""
Preprocessing without Google dictionary adds some variance to our ensemble.
The following 3 steps step can be omitted without major reduction in model performance.
"""
from text_processing_wo_google import *
from feature_extraction1_wo_google import *  
from word2vec_without_google_dict import *




### Modelling from Igor
from generate_feature_importances import *
from generate_models import *
from generate_model_wo_google import *
from generate_ensemble_output_from_models import *
### End of modelling from Igor




#####################################
# START of modelling from Kostia
#Sat, 23 Apr 2016 11:30:30
#Igor's best 0.43854 + kostia best 0.44022 (weights 3 to 1)
#Edit description	submission_kostia + igor final_ensemble (1 to 3 weights).csv	0.43819	0.43704	
#Sat, 23 Apr 2016 11:18:28
#igor: final ensemble version (CV 0.43557) 8 models including xgboost w/o google dict
#Edit description	submission_2016-04-23_ensemble_8models_Igor_final.csv	0.43854	0.43723

"""
Ensemble generation was done by a different method.
This step is included in order to reproduce our submissions.
However, this such modelling might be considered as redundant 
as score improvement is small and such an improvement might be achived within the files
prepared by Igor if the proper feature lists are selected for the models.
"""

"""
modelling was done by the following script which selects random feature subsets
"""
# from ensemble_script_random_version import *
### if the previous script is run, some filenames with model names have to be changed manually
### (The algorithm picks random numbers and then saves files with those number. 
### Then the proper files should be loaded for stacking.)

# the following line reproduces the output which is used in our submissions
from ensemble_script_imitation_version import *


### Selecting models for ensembling
from model_selecting import *
### END of modelling from Kostia
#########################################


"""
The outputs from Igor and Kostia were blended together
with weights 0.75*Igor + 0.25*Kostia
"""





