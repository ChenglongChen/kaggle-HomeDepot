# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: script for testing 1st level model with reg_xgb_tree

"""

import os
import sys

from utils import time_utils

if len(sys.argv) >= 3:
    suffix = sys.argv[1]
    threshold = float(sys.argv[2])
else:
    suffix = time_utils._timestamp_pretty()
    threshold = 0.05

cmd = "python get_feature_conf_nonlinear.py -d 10 -o feature_conf_nonlinear_%s.py"%suffix
os.system(cmd)

cmd = "python feature_combiner.py -l 1 -c feature_conf_nonlinear_%s -n basic_nonlinear_%s -t %.6f"%(suffix, suffix, threshold)
os.system(cmd)

cmd = "python task.py -m single -f basic_nonlinear_%s -l reg_xgb_tree -e 100"%suffix
os.system(cmd)
