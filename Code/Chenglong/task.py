# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: definitions for
        - learner & ensemble learner
        - feature & stacking feature
        - task & stacking task
        - task optimizer

"""

import os
import sys
import time
from optparse import OptionParser

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

import config
from utils import dist_utils, logging_utils, pkl_utils, time_utils
from utils.xgb_utils import XGBRegressor, HomedepotXGBClassifier as XGBClassifier
from utils.rgf_utils import RGFRegressor
from utils.skl_utils import SVR, LinearSVR, KNNRegressor, AdaBoostRegressor, RandomRidge
try:
    from utils.keras_utils import KerasDNNRegressor
except:
    pass
from model_param_space import ModelParamSpace


class Learner:
    def __init__(self, learner_name, param_dict):
        self.learner_name = learner_name
        self.param_dict = param_dict
        self.learner = self._get_learner()

    def __str__(self):
        return self.learner_name

    def _get_learner(self):
        # xgboost
        if self.learner_name in ["reg_xgb_linear", "reg_xgb_tree", "reg_xgb_tree_best_single_model"]:
            return XGBRegressor(**self.param_dict)
        if self.learner_name in ["clf_xgb_linear", "clf_xgb_tree"]:
            return XGBClassifier(**self.param_dict)
        # sklearn
        if self.learner_name == "reg_skl_lasso":
            return Lasso(**self.param_dict)
        if self.learner_name == "reg_skl_ridge":
            return Ridge(**self.param_dict)
        if self.learner_name == "reg_skl_random_ridge":
            return RandomRidge(**self.param_dict)
        if self.learner_name == "reg_skl_bayesian_ridge":
            return BayesianRidge(**self.param_dict)
        if self.learner_name == "reg_skl_svr":
            return SVR(**self.param_dict)
        if self.learner_name == "reg_skl_lsvr":
            return LinearSVR(**self.param_dict)
        if self.learner_name == "reg_skl_knn":
            return KNNRegressor(**self.param_dict)
        if self.learner_name == "reg_skl_etr":
            return ExtraTreesRegressor(**self.param_dict)
        if self.learner_name == "reg_skl_rf":
            return RandomForestRegressor(**self.param_dict)
        if self.learner_name == "reg_skl_gbm":
            return GradientBoostingRegressor(**self.param_dict)
        if self.learner_name == "reg_skl_adaboost":
            return AdaBoostRegressor(**self.param_dict)
        # keras
        if self.learner_name == "reg_keras_dnn":
            try:
                return KerasDNNRegressor(**self.param_dict)
            except:
                return None
        # rgf
        if self.learner_name == "reg_rgf":
            return RGFRegressor(**self.param_dict)
        # ensemble
        if self.learner_name == "reg_ensemble":
            return EnsembleLearner(**self.param_dict)
            
        return None

    def fit(self, X, y, feature_names=None):
        if feature_names is not None:
            self.learner.fit(X, y, feature_names)
        else:
            self.learner.fit(X, y)
        return self

    def predict(self, X, feature_names=None):
        if feature_names is not None:
            y_pred = self.learner.predict(X, feature_names)
        else:
            y_pred = self.learner.predict(X)
        # relevance is in [1,3]
        y_pred = np.clip(y_pred, 1., 3.)
        return y_pred

    def plot_importance(self):
        ax = self.learner.plot_importance()
        return ax


class EnsembleLearner:
    def __init__(self, learner_dict):
        self.learner_dict = learner_dict

    def __str__(self):
        return "EnsembleLearner"

    def fit(self, X, y):
        for learner_name in self.learner_dict.keys():
            p = self.learner_dict[learner_name]["param"]
            l = Learner(learner_name, p)._get_learner()
            if l is not None:
                self.learner_dict[learner_name]["learner"] = l.fit(X, y)
            else:
                self.learner_dict[learner_name]["learner"] = None
        return self

    def predict(self, X):
        y_pred = np.zeros((X.shape[0]), dtype=float)
        w_sum = 0.
        for learner_name in self.learner_dict.keys():
            l = self.learner_dict[learner_name]["learner"]
            if l is not None:
                w = self.learner_dict[learner_name]["weight"]
                y_pred += w * l.predict(X)
                w_sum += w
        y_pred /= w_sum
        return y_pred


class Feature:
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.data_dict = self._load_data_dict()
        self.splitter = self.data_dict["splitter"]
        self.n_iter = self.data_dict["n_iter"]

    def __str__(self):
        return self.feature_name

    def _load_data_dict(self):
        fname = os.path.join(config.FEAT_DIR+"/Combine", self.feature_name+config.FEAT_FILE_SUFFIX)
        data_dict = pkl_utils._load(fname)
        return data_dict

    ## for CV
    def _get_train_valid_data(self, i):
        # feature
        X_basic_train = self.data_dict["X_train_basic"][self.splitter[i][0], :]
        X_basic_valid = self.data_dict["X_train_basic"][self.splitter[i][1], :]
        if self.data_dict["basic_only"]:
            X_train, X_valid = X_basic_train, X_basic_valid
        else:
            X_train_cv = self.data_dict["X_train_cv"][self.splitter[i][0], :, i]
            X_valid_cv = self.data_dict["X_train_cv"][self.splitter[i][1], :, i]
            X_train = np.hstack((X_basic_train, X_train_cv))
            X_valid = np.hstack((X_basic_valid, X_valid_cv))
        # label
        y_train = self.data_dict["y_train"][self.splitter[i][0]]
        y_valid = self.data_dict["y_train"][self.splitter[i][1]]

        return X_train, y_train, X_valid, y_valid

    ## for refit
    def _get_train_test_data(self):
        # feature
        X_basic = self.data_dict["X_train_basic"]
        if self.data_dict["basic_only"]:
            X_train = X_basic
        else:
            X_train = np.hstack((X_basic, self.data_dict["X_train_cv_all"]))
        # test already contains cv features if any
        # in the order [basic, cv]
        X_test = self.data_dict["X_test"]
        # label
        y_train = self.data_dict["y_train"]

        return X_train, y_train, X_test

    ## for feature importance
    def _get_feature_names(self):
        return self.data_dict["feature_names"]


class StackingFeature(Feature):
    def __init__(self, feature_name):
        super().__init__(feature_name)
        self.splitter_prev = self.data_dict["splitter_prev"]

    ## for CV
    def _get_train_valid_data(self, i):
        # feature
        X_train_cv = self.data_dict["X_train_cv"][i][self.splitter[i][0]]
        X_valid_cv = self.data_dict["X_train_cv"][i][self.splitter[i][1]]
        if self.data_dict["has_basic"]:
            X_train_basic = self.data_dict["X_train_basic"][self.splitter_prev[i]]
            X_train = np.hstack([X_train_basic[self.splitter[i][0]], X_train_cv])
            X_valid = np.hstack([X_train_basic[self.splitter[i][1]], X_valid_cv])
        # label
        y_train = self.data_dict["y_train_cv"][i][self.splitter[i][0]]
        y_valid = self.data_dict["y_train_cv"][i][self.splitter[i][1]]

        return X_train, y_train, X_valid, y_valid

    ## for refit
    def _get_train_test_data(self, i):
        # feature
        X_train = self.data_dict["X_train_cv"][i]
        if self.data_dict["has_basic"]:
            X_train_basic = self.data_dict["X_train_basic"][self.splitter_prev[i]]
            X_train = np.hstack([X_train_basic, X_train])
        # test already contains basic features if any
        # in the order [basic, cv]
        X_test = self.data_dict["X_test"]
        # label
        y_train = self.data_dict["y_train_cv"][i]

        return X_train, y_train, X_test


class Task:
    def __init__(self, learner, feature, suffix, logger, verbose=True, plot_importance=False):
        self.learner = learner
        self.feature = feature
        self.suffix = suffix
        self.logger = logger
        self.verbose = verbose
        self.plot_importance = plot_importance
        self.n_iter = self.feature.n_iter
        self.rmse_cv_mean = 0
        self.rmse_cv_std = 0

    def __str__(self):
        return "[Feat@%s]_[Learner@%s]%s"%(str(self.feature), str(self.learner), str(self.suffix))

    def _print_param_dict(self, d, prefix="      ", incr_prefix="      "):
        for k,v in sorted(d.items()):
            if isinstance(v, dict):
                self.logger.info("%s%s:" % (prefix,k))
                self._print_param_dict(v, prefix+incr_prefix, incr_prefix)
            else:
                self.logger.info("%s%s: %s" % (prefix,k,v))

    def cv(self):
        start = time.time()
        if self.verbose:
            self.logger.info("="*50)
            self.logger.info("Task")
            self.logger.info("      %s" % str(self.__str__()))
            self.logger.info("Param")
            self._print_param_dict(self.learner.param_dict)
            self.logger.info("Result")
            self.logger.info("      Run      RMSE        Shape")
    
        rmse_cv = np.zeros(self.n_iter)
        for i in range(self.n_iter):
            # data
            X_train, y_train, X_valid, y_valid = self.feature._get_train_valid_data(i)
            # fit
            self.learner.fit(X_train, y_train)
            y_pred = self.learner.predict(X_valid)
            rmse_cv[i] = dist_utils._rmse(y_valid, y_pred)
            # log
            self.logger.info("      {:>3}    {:>8}    {} x {}".format(
                i+1, np.round(rmse_cv[i],6), X_train.shape[0], X_train.shape[1]))
            # save
            fname = "%s/Run%d/valid.pred.%s.csv"%(config.OUTPUT_DIR, i+1, self.__str__())
            df = pd.DataFrame({"target": y_valid, "prediction": y_pred})
            df.to_csv(fname, index=False, columns=["target", "prediction"])
            if hasattr(self.learner.learner, "predict_proba"):
                y_proba = self.learner.learner.predict_proba(X_valid)
                fname = "%s/Run%d/valid.proba.%s.csv"%(config.OUTPUT_DIR, i+1, self.__str__())
                columns = ["proba%d"%i for i in range(y_proba.shape[1])]
                df = pd.DataFrame(y_proba, columns=columns)
                df["target"] = y_valid
                df.to_csv(fname, index=False)

        self.rmse_cv_mean = np.mean(rmse_cv)
        self.rmse_cv_std = np.std(rmse_cv)
        end = time.time()
        _sec = end - start
        _min = int(_sec/60.)
        if self.verbose:
            self.logger.info("RMSE")
            self.logger.info("      Mean: %.6f"%self.rmse_cv_mean)
            self.logger.info("      Std: %.6f"%self.rmse_cv_std)
            self.logger.info("Time")
            if _min > 0:
                self.logger.info("      %d mins"%_min)
            else:
                self.logger.info("      %d secs"%_sec)
            self.logger.info("-"*50)
        return self

    def refit(self):
        X_train, y_train, X_test = self.feature._get_train_test_data()
        if self.plot_importance:
            feature_names = self.feature._get_feature_names()
            self.learner.fit(X_train, y_train, feature_names)
            y_pred = self.learner.predict(X_test, feature_names)
        else:
            self.learner.fit(X_train, y_train)
            y_pred = self.learner.predict(X_test)
        
        id_test = self.feature.data_dict["id_test"].astype(int)

        # save
        fname = "%s/%s/test.pred.%s.csv"%(config.OUTPUT_DIR, "All", self.__str__())
        pd.DataFrame({"id": id_test, "prediction": y_pred}).to_csv(fname, index=False)
        if hasattr(self.learner.learner, "predict_proba"):
            if self.plot_importance:
                feature_names = self.feature._get_feature_names()
                y_proba = self.learner.learner.predict_proba(X_test, feature_names)
            else:
                y_proba = self.learner.learner.predict_proba(X_test)
            fname = "%s/%s/test.proba.%s.csv"%(config.OUTPUT_DIR, "All", self.__str__())
            columns = ["proba%d"%i for i in range(y_proba.shape[1])]
            df = pd.DataFrame(y_proba, columns=columns)
            df["id"] = id_test
            df.to_csv(fname, index=False)

        # submission
        fname = "%s/test.pred.%s.[Mean%.6f]_[Std%.6f].csv"%(
            config.SUBM_DIR, self.__str__(), self.rmse_cv_mean, self.rmse_cv_std)
        pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(fname, index=False)

        # plot importance
        if self.plot_importance:
            ax = self.learner.plot_importance()
            ax.figure.savefig("%s/%s.pdf"%(config.FIG_DIR, self.__str__()))
        return self

    def go(self):
        self.cv()
        self.refit()
        return self


class StackingTask(Task):
    def __init__(self, learner, feature, suffix, logger, verbose=True, refit_once=False):
        super().__init__(learner, feature, suffix, logger, verbose)
        self.refit_once = refit_once

    def refit(self):
        for i in range(self.n_iter):
            if self.refit_once and i >= 1:
                break
            X_train, y_train, X_test = self.feature._get_train_test_data(i)
            self.learner.fit(X_train, y_train)
            if i == 0:
                y_pred = self.learner.predict(X_test)
                if hasattr(self.learner.learner, "predict_proba"):
                    y_proba = self.learner.learner.predict_proba(X_test)
            else:
                y_pred += self.learner.predict(X_test)
                if hasattr(self.learner.learner, "predict_proba"):
                    y_proba += self.learner.learner.predict_proba(X_test)
        if not self.refit_once:
            y_pred /= float(self.n_iter)
            if hasattr(self.learner.learner, "predict_proba"):
                y_proba /= float(self.n_iter)

        id_test = self.feature.data_dict["id_test"].astype(int)

        # save
        fname = "%s/%s/test.pred.%s.csv"%(config.OUTPUT_DIR, "All", self.__str__())
        pd.DataFrame({"id": id_test, "prediction": y_pred}).to_csv(fname, index=False)
        if hasattr(self.learner.learner, "predict_proba"):
            fname = "%s/%s/test.proba.%s.csv"%(config.OUTPUT_DIR, "All", self.__str__())
            columns = ["proba%d"%i for i in range(y_proba.shape[1])]
            print(y_proba.shape)
            print(len(columns))
            df = pd.DataFrame(y_proba, columns=columns)
            df["id"] = id_test
            df.to_csv(fname, index=False)

        # submission
        fname = "%s/test.pred.%s.[Mean%.6f]_[Std%.6f].csv"%(
            config.SUBM_DIR, self.__str__(), self.rmse_cv_mean, self.rmse_cv_std)
        pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(fname, index=False)
        return self


class TaskOptimizer:
    def __init__(self, task_mode, learner_name, feature_name, logger, 
                    max_evals=100, verbose=True, refit_once=False, plot_importance=False):
        self.task_mode = task_mode
        self.learner_name = learner_name
        self.feature_name = feature_name
        self.feature = self._get_feature()
        self.logger = logger
        self.max_evals = max_evals
        self.verbose = verbose
        self.refit_once = refit_once
        self.plot_importance = plot_importance
        self.trial_counter = 0
        self.model_param_space = ModelParamSpace(self.learner_name)

    def _get_feature(self):
        if self.task_mode == "single":
            feature = Feature(self.feature_name)
        elif self.task_mode == "stacking":
            feature = StackingFeature(self.feature_name)
        return feature

    def _obj(self, param_dict):
        self.trial_counter += 1
        param_dict = self.model_param_space._convert_int_param(param_dict)
        learner = Learner(self.learner_name, param_dict)
        suffix = "_[Id@%s]"%str(self.trial_counter)
        if self.task_mode == "single":
            self.task = Task(learner, self.feature, suffix, self.logger, self.verbose, self.plot_importance)
        elif self.task_mode == "stacking":
            self.task = StackingTask(learner, self.feature, suffix, self.logger, self.verbose, self.refit_once)
        self.task.go()
        ret = {
            "loss": self.task.rmse_cv_mean,
            "attachments": {
                "std": self.task.rmse_cv_std,
            },
            "status": STATUS_OK,
        }
        return ret

    def run(self):
        start = time.time()
        trials = Trials()
        best = fmin(self._obj, self.model_param_space._build_space(), tpe.suggest, self.max_evals, trials)
        best_params = space_eval(self.model_param_space._build_space(), best)
        best_params = self.model_param_space._convert_int_param(best_params)
        trial_rmses = np.asarray(trials.losses(), dtype=float)
        best_ind = np.argmin(trial_rmses)
        best_rmse_mean = trial_rmses[best_ind]
        best_rmse_std = trials.trial_attachments(trials.trials[best_ind])["std"]
        self.logger.info("-"*50)
        self.logger.info("Best RMSE")
        self.logger.info("      Mean: %.6f"%best_rmse_mean)
        self.logger.info("      std: %.6f"%best_rmse_std)
        self.logger.info("Best param")
        self.task._print_param_dict(best_params)
        end = time.time()
        _sec = end - start
        _min = int(_sec/60.)
        self.logger.info("Time")
        if _min > 0:
            self.logger.info("      %d mins"%_min)
        else:
            self.logger.info("      %d secs"%_sec)
        self.logger.info("-"*50)


#------------------------ Main -------------------------
def main(options):
    logname = "[Feat@%s]_[Learner@%s]_hyperopt_%s.log"%(
        options.feature_name, options.learner_name, time_utils._timestamp())
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    optimizer = TaskOptimizer(options.task_mode, options.learner_name, 
        options.feature_name, logger, options.max_evals, verbose=True, 
        refit_once=options.refit_once, plot_importance=options.plot_importance)
    optimizer.run()

def parse_args(parser):
    parser.add_option("-m", "--mode", type="string", dest="task_mode",
        help="task mode", default="single")
    parser.add_option("-f", "--feat", type="string", dest="feature_name",
        help="feature name", default="basic")
    parser.add_option("-l", "--learner", type="string", dest="learner_name", 
        help="learner name", default="reg_skl_ridge")
    parser.add_option("-e", "--eval", type="int", dest="max_evals", 
        help="maximun number of evals for hyperopt", default=100)
    parser.add_option("-o", default=False, action="store_true", dest="refit_once",
        help="stacking refit_once")
    parser.add_option("-p", default=False, action="store_true", dest="plot_importance",
        help="plot feautre importance (currently only for xgboost)")

    (options, args) = parser.parse_args()
    return options, args


if __name__ == "__main__":

    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
