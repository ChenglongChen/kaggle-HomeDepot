# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for RGF models

"""

import os
import sys

import numpy as np

from . import os_utils
sys.path.append("..")
import config


class RGFRegressor:
    def __init__(self, reg_L2=0.1, reg_sL2=0.0001, max_leaf_forest=10000, num_iteration_opt=10,
                num_tree_search=1, min_pop=10, opt_interval=100, opt_stepsize=0.5):

        self.param = {
            "reg_L2": reg_L2,
            "reg_sL2": reg_sL2,
            "max_leaf_forest": max_leaf_forest,
            "num_iteration_opt": num_iteration_opt,
            "num_tree_search": num_tree_search,
            "min_pop": min_pop,
            "opt_interval": opt_interval,
            "opt_stepsize": opt_stepsize,
        }

        # create tmp dir to hold data and model (especially the latter)
        self.tmp_dir = "%s/%s"%(config.TMP_DIR, os_utils._gen_signature())
        os_utils._create_dirs([self.tmp_dir])
        self.model_fn_prefix = "%s/rgf_model"%self.tmp_dir
            
    def __del__(self):
        ## delete tmp dir
        os_utils._remove_dirs([self.tmp_dir])

    def __str__(self):
        return "RGFRegressor"

    def fit(self, X, y):

        # write train data to file
        train_x_fn = "%s/data.x"%self.tmp_dir
        train_y_fn = "%s/data.y"%self.tmp_dir
        np.savetxt(train_x_fn, X, fmt="%.6f", delimiter="\t")
        np.savetxt(train_y_fn, y, fmt="%.6f", delimiter="\t")

        ## write train param to file
        params = [
            "train_x_fn=",train_x_fn,"\n",
            "train_y_fn=",train_y_fn,"\n",
            #"train_w_fn=",weight_train_path,"\n",
            "model_fn_prefix=",self.model_fn_prefix,"\n",
            "reg_L2=", self.param["reg_L2"], "\n",
            "reg_sL2=", self.param["reg_sL2"], "\n",
            #"reg_depth=", 1.01, "\n",
            "algorithm=","RGF","\n",
            "loss=","LS","\n",
            #"opt_interval=", 100, "\n",
            # save model at the end of training
            "test_interval=", self.param["max_leaf_forest"],"\n", 
            "max_leaf_forest=", self.param["max_leaf_forest"],"\n",
            "num_iteration_opt=", self.param["num_iteration_opt"], "\n",
            "num_tree_search=", self.param["num_tree_search"], "\n",
            "min_pop=", self.param["min_pop"], "\n",
            "opt_interval=", self.param["opt_interval"], "\n",
            "opt_stepsize=", self.param["opt_stepsize"], "\n",
            "NormalizeTarget"
        ]
        params = "".join([str(p) for p in params])

        rgf_setting = "%s/rgf_setting"%self.tmp_dir # DOES NOT contain ".inp"
        with open(rgf_setting+".inp", "w") as f:
            f.write(params)

        ## train rgf
        rgf_log = "%s/rgf_log"%self.tmp_dir
        cmd = "perl %s %s train %s >> %s"%(
                config.RGF_CALL_EXE, config.RGF_EXE, rgf_setting, rgf_log)
        os.system(cmd)

        return self

    def predict(self, X):

        ## write data to file
        valid_x_fn = "%s/data.x"%self.tmp_dir
        valid_y_fn = "%s/data.y"%self.tmp_dir
        np.savetxt(valid_x_fn, X, fmt="%.6f", delimiter="\t")

        ## write predict params to file
        model_fn = self.model_fn_prefix + "-01"
        params = [
            "test_x_fn=", valid_x_fn,"\n",
            "model_fn=", model_fn,"\n",
            "prediction_fn=", valid_y_fn
        ]
        params = "".join([str(p) for p in params])
        
        rgf_setting = "%s/rgf_setting"%self.tmp_dir
        with open(rgf_setting+".inp", "w") as f:
            f.write(params)

        ## predict
        rgf_log = "%s/rgf_log"%self.tmp_dir
        cmd = "perl %s %s predict %s >> %s"%(
                config.RGF_CALL_EXE, config.RGF_EXE, rgf_setting, rgf_log)
        os.system(cmd)

        y_pred = np.loadtxt(valid_y_fn, dtype=float)

        return y_pred
