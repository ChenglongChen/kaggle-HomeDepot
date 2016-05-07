# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: script for generating the best ensemble model from Chenglong's side
@note: 1. make sure you have run `python run_data.py` first
	   2. make sure you have built `some diverse` 1st level models first (see `./Log/level1_models` for example)

"""

import os


cmd = "python run_stacking_ridge.py -l 2 -d 0 -t 10 -c 1 -L reg_ensemble -o"
os.system(cmd)
