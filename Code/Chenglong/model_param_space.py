# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: model parameter space

"""

import numpy as np
from hyperopt import hp

import config


## xgboost
xgb_random_seed = config.RANDOM_SEED
xgb_nthread = config.NUM_CORES
xgb_n_estimators_min = 100
xgb_n_estimators_max = 1000
xgb_n_estimators_step = 10

## sklearn
skl_random_seed = config.RANDOM_SEED
skl_n_jobs = config.NUM_CORES
skl_n_estimators_min = 100
skl_n_estimators_max = 1000
skl_n_estimators_step = 10


# ---------------------------- XGBoost ---------------------------------------
## regression with linear booster
param_space_reg_xgb_linear = {
    "booster": "gblinear",
    "objective": "reg:linear",
    "base_score": config.BASE_SCORE,
    "n_estimators" : hp.quniform("n_estimators", xgb_n_estimators_min, xgb_n_estimators_max, xgb_n_estimators_step),
    "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "reg_alpha" : hp.loguniform("reg_alpha", np.log(1e-10), np.log(1e1)),
    "reg_lambda" : hp.loguniform("reg_lambda", np.log(1e-10), np.log(1e1)),
    "reg_lambda_bias" : hp.quniform("reg_lambda_bias", 0, 3, 0.1),
    "nthread": xgb_nthread,
    "seed": xgb_random_seed,
}

## regression with tree booster
param_space_reg_xgb_tree = {
    "booster": "gbtree",
    "objective": "reg:linear",
    "base_score": config.BASE_SCORE,
    "n_estimators" : hp.quniform("n_estimators", xgb_n_estimators_min, xgb_n_estimators_max, xgb_n_estimators_step),
    "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "gamma": hp.loguniform("gamma", np.log(1e-10), np.log(1e1)),
    "reg_alpha" : hp.loguniform("reg_alpha", np.log(1e-10), np.log(1e1)),
    "reg_lambda" : hp.loguniform("reg_lambda", np.log(1e-10), np.log(1e1)),
    "min_child_weight": hp.loguniform("min_child_weight", np.log(1e-10), np.log(1e2)),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "subsample": hp.quniform("subsample", 0.1, 1, 0.05),
    "colsample_bytree": 1,
    "colsample_bylevel": hp.quniform("colsample_bylevel", 0.1, 1, 0.05),
    "nthread": xgb_nthread,
    "seed": xgb_random_seed,
}

## regression with tree booster (parm for best single model)
param_space_reg_xgb_tree_best_single_model = {
    "booster": "gbtree",
    "objective": "reg:linear",
    "base_score": config.BASE_SCORE,
    "n_estimators" : 880,
    "learning_rate" : 0.014,
    "gamma": 0.0007375692431738125,
    "reg_alpha" : 0.0024595063900801238,
    "reg_lambda" : 0.0031003919409140915,
    "min_child_weight": 96.14430369781684,
    "max_depth": 7,
    "subsample": 0.8500000000000001,
    "colsample_bytree": 1,
    "colsample_bylevel": 0.15000000000000002,
    "nthread": xgb_nthread,
    "seed": xgb_random_seed,
}

## classification with tree booster
param_space_clf_xgb_tree = {
    "booster": "gbtree",
    "objective": "multi:softprob",
    "base_score": config.BASE_SCORE,
    "n_estimators" : hp.quniform("n_estimators", xgb_n_estimators_min, xgb_n_estimators_max, xgb_n_estimators_step),
    "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "gamma": hp.loguniform("gamma", np.log(1e-10), np.log(1e1)),
    "reg_alpha" : hp.loguniform("reg_alpha", np.log(1e-10), np.log(1e1)),
    "reg_lambda" : hp.loguniform("reg_lambda", np.log(1e-10), np.log(1e1)),
    "min_child_weight": hp.loguniform("min_child_weight", np.log(1e-10), np.log(1e2)),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "subsample": hp.quniform("subsample", 0.1, 1, 0.05),
    "colsample_bytree": 1,
    "colsample_bylevel": hp.quniform("colsample_bylevel", 0.1, 1, 0.05),
    "nthread": xgb_nthread,
    "seed": xgb_random_seed,
}

# -------------------------------------- Sklearn ---------------------------------------------
## lasso
param_space_reg_skl_lasso = {
    "alpha": hp.loguniform("alpha", np.log(0.00001), np.log(0.1)),
    "normalize": hp.choice("normalize", [True, False]),
    "random_state": skl_random_seed
}

## ridge regression
param_space_reg_skl_ridge = {
    "alpha": hp.loguniform("alpha", np.log(0.01), np.log(20)),
    "normalize": hp.choice("normalize", [True, False]),
    "random_state": skl_random_seed
}

## Bayesian Ridge Regression
param_space_reg_skl_bayesian_ridge = {
    "alpha_1": hp.loguniform("alpha_1", np.log(1e-10), np.log(1e2)),
    "alpha_2": hp.loguniform("alpha_2", np.log(1e-10), np.log(1e2)),
    "lambda_1": hp.loguniform("lambda_1", np.log(1e-10), np.log(1e2)),
    "lambda_2": hp.loguniform("lambda_2", np.log(1e-10), np.log(1e2)),
    "normalize": hp.choice("normalize", [True, False])
}

## random ridge regression
param_space_reg_skl_random_ridge = {
    "alpha": hp.loguniform("alpha", np.log(0.01), np.log(20)),
    "normalize": hp.choice("normalize", [True, False]),
    "poly": hp.choice("poly", [False]),
    "n_estimators": hp.quniform("n_estimators", 2, 50, 2),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "bootstrap": hp.choice("bootstrap", [True, False]),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
    "random_state": skl_random_seed
}

## linear support vector regression
param_space_reg_skl_lsvr = {
    "normalize": hp.choice("normalize", [True, False]),
    "C": hp.loguniform("C", np.log(1), np.log(100)),
    "epsilon": hp.loguniform("epsilon", np.log(0.001), np.log(0.1)),    
    "loss": hp.choice("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]),
    "random_state": skl_random_seed,
}

## support vector regression
param_space_reg_skl_svr = {
    "normalize": hp.choice("normalize", [True]),
    "C": hp.loguniform("C", np.log(1), np.log(1)),
    "gamma": hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
    "degree": hp.quniform("degree", 1, 3, 1),
    "epsilon": hp.loguniform("epsilon", np.log(0.001), np.log(0.1)),    
    "kernel": hp.choice("kernel", ["rbf", "poly"])
}

## K Nearest Neighbors Regression
param_space_reg_skl_knn = {
    "normalize": hp.choice("normalize", [True, False]),
    "n_neighbors": hp.quniform("n_neighbors", 1, 20, 1),
    "weights": hp.choice("weights", ["uniform", "distance"]),
    "leaf_size": hp.quniform("leaf_size", 10, 100, 10),
    "metric": hp.choice("metric", ["cosine", "minkowski"][1:]),
}

## extra trees regressor
param_space_reg_skl_etr = {
    "n_estimators": hp.quniform("skl_etr__n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
    "max_features": hp.quniform("skl_etr__max_features", 0.1, 1, 0.05),
    "min_samples_split": hp.quniform("skl_etr__min_samples_split", 1, 15, 1),
    "min_samples_leaf": hp.quniform("skl_etr__min_samples_leaf", 1, 15, 1),
    "max_depth": hp.quniform("skl_etr__max_depth", 1, 10, 1),
    "random_state": skl_random_seed,
    "n_jobs": skl_n_jobs,
    "verbose": 0,
}

## random forest regressor
param_space_reg_skl_rf = {
    "n_estimators": hp.quniform("skl_rf__n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
    "max_features": hp.quniform("skl_rf__max_features", 0.1, 1, 0.05),
    "min_samples_split": hp.quniform("skl_rf__min_samples_split", 1, 15, 1),
    "min_samples_leaf": hp.quniform("skl_rf__min_samples_leaf", 1, 15, 1),
    "max_depth": hp.quniform("skl_rf__max_depth", 1, 10, 1),
    "random_state": skl_random_seed,
    "n_jobs": skl_n_jobs,
    "verbose": 0,
}

## gradient boosting regressor
param_space_reg_skl_gbm = {
    "n_estimators": hp.quniform("skl_gbm__n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
    "learning_rate" : hp.qloguniform("skl__gbm_learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "max_features": hp.quniform("skl_gbm__max_features", 0.1, 1, 0.05),
    "max_depth": hp.quniform("skl_gbm__max_depth", 1, 10, 1),
    "min_samples_leaf": hp.quniform("skl_gbm__min_samples_leaf", 1, 15, 1),
    "random_state": skl_random_seed,
    "verbose": 0,
}

## adaboost regressor
param_space_reg_skl_adaboost = {
    "base_estimator": hp.choice("base_estimator", ["dtr", "etr"]),
    "n_estimators": hp.quniform("n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
    "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "loss": hp.choice("loss", ["linear", "square", "exponential"]),
    "random_state": skl_random_seed,
}

# -------------------------------------- Keras ---------------------------------------------
## regression with Keras' deep neural network
param_space_reg_keras_dnn = {
    "input_dropout": hp.quniform("input_dropout", 0, 0.2, 0.05),
    "hidden_layers": hp.quniform("hidden_layers", 1, 5, 1),
    "hidden_units": hp.quniform("hidden_units", 32, 256, 32),
    "hidden_activation": hp.choice("hidden_activation", ["prelu", "relu", "elu"]),
    "hidden_dropout": hp.quniform("hidden_dropout", 0, 0.5, 0.05),
    "batch_norm": hp.choice("batch_norm", ["before_act", "after_act", "no"]),
    "optimizer": hp.choice("optimizer", ["adam", "adadelta", "rmsprop"]),
    "batch_size": hp.quniform("batch_size", 16, 128, 16),
    "nb_epoch": hp.quniform("nb_epoch", 1, 20, 1),
}

# -------------------------------------- RGF ---------------------------------------------
param_space_reg_rgf = {
    "reg_L2": hp.loguniform("reg_L2", np.log(0.1), np.log(10)),
    "reg_sL2": hp.loguniform("reg_sL2", np.log(0.00001), np.log(0.1)),
    "max_leaf_forest": hp.quniform("max_leaf_forest", 10, 1000, 10),
    "num_iteration_opt": hp.quniform("num_iteration_opt", 5, 20, 1),
    "num_tree_search": hp.quniform("num_tree_search", 1, 10, 1),
    "min_pop": hp.quniform("min_pop", 1, 20, 1),
    "opt_interval": hp.quniform("opt_interval", 10, 200, 10),
    "opt_stepsize": hp.quniform("opt_stepsize", 0.1, 1.0, 0.1)
}

# -------------------------------------- Ensemble ---------------------------------------------
# 1. The following learners are chosen to build ensemble for their fast learning speed.
# 2. In our final submission, we used fix weights. 
#    However, you can also try to optimize the ensemble weights in the meantime.
param_space_reg_ensemble = {
    # 1. fix weights (used in final submission)
    "learner_dict": {
        "reg_skl_ridge": {
            "param": param_space_reg_skl_ridge,
            "weight": 4.0,
        },
        "reg_keras_dnn": {
            "param": param_space_reg_keras_dnn,
            "weight": 1.0,
        }, 
        "reg_xgb_tree": {
            "param": param_space_reg_xgb_tree,
            "weight": 1.0,
        }, 
        "reg_skl_etr": {
            "param": param_space_reg_skl_etr,
            "weight": 1.0,
        }, 
        "reg_skl_rf": {
            "param": param_space_reg_skl_rf,
            "weight": 1.0,
        }, 
    },
    # # 2. optimizing weights
    # "learner_dict": {
    #     "reg_skl_ridge": {
    #         "param": param_space_reg_skl_ridge,
    #         "weight": hp.quniform("reg_skl_ridge__weight", 1.0, 1.0, 0.1), # fix this one
    #     },
    #     "reg_keras_dnn": {
    #         "param": param_space_reg_keras_dnn,
    #         "weight": hp.quniform("reg_keras_dnn__weight", 0.0, 1.0, 0.1),
    #     }, 
    #     "reg_xgb_tree": {
    #         "param": param_space_reg_xgb_tree,
    #         "weight": hp.quniform("reg_xgb_tree__weight", 0.0, 1.0, 0.1),
    #     }, 
    #     "reg_skl_etr": {
    #         "param": param_space_reg_skl_etr,
    #         "weight": hp.quniform("reg_skl_etr__weight", 0.0, 1.0, 0.1),
    #     }, 
    #     "reg_skl_rf": {
    #         "param": param_space_reg_skl_rf,
    #         "weight": hp.quniform("reg_skl_rf__weight", 0.0, 1.0, 0.1),
    #     }, 
    # },
}

# -------------------------------------- All ---------------------------------------------
param_space_dict = {
    # xgboost
    "reg_xgb_tree": param_space_reg_xgb_tree,
    "reg_xgb_tree_best_single_model": param_space_reg_xgb_tree_best_single_model,
    "reg_xgb_linear": param_space_reg_xgb_linear,
    "clf_xgb_tree": param_space_clf_xgb_tree,
    # sklearn
    "reg_skl_lasso": param_space_reg_skl_lasso,
    "reg_skl_ridge": param_space_reg_skl_ridge,
    "reg_skl_bayesian_ridge": param_space_reg_skl_bayesian_ridge,
    "reg_skl_random_ridge": param_space_reg_skl_random_ridge,
    "reg_skl_lsvr": param_space_reg_skl_lsvr,
    "reg_skl_svr": param_space_reg_skl_svr,
    "reg_skl_knn": param_space_reg_skl_knn,
    "reg_skl_etr": param_space_reg_skl_etr,
    "reg_skl_rf": param_space_reg_skl_rf,
    "reg_skl_gbm": param_space_reg_skl_gbm,
    "reg_skl_adaboost": param_space_reg_skl_adaboost,
    # keras
    "reg_keras_dnn": param_space_reg_keras_dnn,
    # rgf
    "reg_rgf": param_space_reg_rgf,
    # ensemble
    "reg_ensemble": param_space_reg_ensemble,
}

int_params = [
    "num_round", "n_estimators", "min_samples_split", "min_samples_leaf",
    "n_neighbors", "leaf_size", "seed", "random_state", "max_depth", "degree",
    "hidden_units", "hidden_layers", "batch_size", "nb_epoch", "dim", "iter", 
    "factor", "iteration", "n_jobs", "max_leaf_forest", "num_iteration_opt", 
    "num_tree_search", "min_pop", "opt_interval",
]
int_params = set(int_params)


class ModelParamSpace:
    def __init__(self, learner_name):
        s = "Wrong learner_name, " + \
            "see model_param_space.py for all available learners."
        assert learner_name in param_space_dict, s
        self.learner_name = learner_name

    def _build_space(self):
        return param_space_dict[self.learner_name]

    def _convert_int_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k,v in param_dict.items():
                if k in int_params:
                    param_dict[k] = int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_int_param(v[i])
                elif isinstance(v, dict):
                    self._convert_int_param(v)
        return param_dict
