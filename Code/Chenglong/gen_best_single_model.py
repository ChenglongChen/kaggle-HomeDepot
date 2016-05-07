# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: script for generating the best single model from Chenglong's side
@note: 1. make sure you have run `python run_data.py` first
	   2. RMSE should be something around 0.438 ~ 0.439

"""

import os


suffix = '201604210409'
threshold = 0.05

cmd = "python feature_combiner.py -l 1 -c feature_conf_nonlinear_%s -n basic_nonlinear_%s -t %.6f"%(suffix, suffix, threshold)
os.system(cmd)

cmd = "python task.py -m single -f basic_nonlinear_%s -l reg_xgb_tree_best_single_model -e 1"%suffix
os.system(cmd)
