# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: extreme ensemble selection module
@note: 
        - support multiprocessing (set multiprocessing = True and multiprocessing_num_cores = #cores)
        - support random weight in greedy forward model selection (set weight_opt_max_evals = 1)
"""

"""
Implement (extreme) ensemble selection with the following techniques described in [1][2]:
1. Ensemble Initialization
2. Bagged Ensemble Selection
3. Ensemble Selection with Replacement
4. Cross-Validated Ensemble Selection
5. Random Weight in Greedy Forward Model Selection

Reference
[1] Ensemble selection from libraries of models, R. Caruana, A. Niculescu-Mizil, G. Crew, and A. Ksikes.
[2] Getting the Most Out of Ensemble Selection, Rich Caruana, Art Munson, and Alexandru Niculescu-Mizil

"""

import csv
import time
from optparse import OptionParser

import scipy
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils.fixes import bincount
from sklearn.cross_validation import check_random_state
from sklearn.cross_validation import BaseShuffleSplit, ShuffleSplit, StratifiedShuffleSplit
# http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
# The solution for me was to add the following code in a place 
# that gets read before any other pylab/matplotlib/pyplot import:
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from utils import dist_utils, np_utils
from utils import logging_utils, os_utils, pkl_utils, time_utils
from get_stacking_feature_conf import get_model_list


splitter_level1 = pkl_utils._load("%s/splits_level1.pkl"%config.SPLIT_DIR)
splitter_level2 = pkl_utils._load("%s/splits_level2.pkl"%config.SPLIT_DIR)
splitter_level3 = pkl_utils._load("%s/splits_level3.pkl"%config.SPLIT_DIR)
assert len(splitter_level1) == len(splitter_level2)
assert len(splitter_level1) == len(splitter_level3)
n_iter = len(splitter_level1)


class StratifiedShuffleSplitReplacement(BaseShuffleSplit):

    def __init__(self, y, n_iter=10, test_size=0.1, train_size=None,
                 random_state=None):

        super(StratifiedShuffleSplitReplacement, self).__init__(
            len(y), n_iter, test_size, train_size, random_state)

        self.y = np.array(y)
        self.classes, self.y_indices = np.unique(y, return_inverse=True)
        n_cls = self.classes.shape[0]

    def _iter_indices(self):
        rng = np.random.RandomState(self.random_state)
        cls_count = bincount(self.y_indices)

        for n in range(self.n_iter):
            train = []
            test = []

            for i, cls in enumerate(self.classes):
                sample_size = int(cls_count[i]*(1-self.test_size))
                randint = rng.randint(cls_count[i], size=sample_size)
                aidx = np.where((self.y == cls))[0]
                iidx = aidx[randint]
                oidx = aidx[list(set(range(cls_count[i])).difference(set(randint)))]

                train.extend(iidx)
                test.extend(oidx)

            train = rng.permutation(train)
            test = rng.permutation(test)

            yield train, test

    def __repr__(self):
        return ('%s(labels=%s, n_iter=%d, test_size=%s, '
                'random_state=%s)' % (
                    self.__class__.__name__,
                    self.y,
                    self.n_iter,
                    str(self.test_size),
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iter


class ExtremeEnsembleSelection:
    def __init__(self, model_folder, model_list, subm_prefix, 
                weight_opt_max_evals=10, w_min=-1., w_max=1., 
                inst_subsample=0.5, inst_subsample_replacement=False, 
                inst_splitter=None,
                model_subsample=1.0, model_subsample_replacement=True,
                bagging_size=10, init_top_k=5, epsilon=0.00001, 
                multiprocessing=False, multiprocessing_num_cores=1,
                enable_extreme=True, random_seed=0):

        self.model_folder = model_folder
        self.model_list = model_list
        self.subm_prefix = subm_prefix
        self.weight_opt_max_evals = weight_opt_max_evals
        self.w_min = w_min
        self.w_max = w_max
        assert inst_subsample > 0 and inst_subsample <= 1.
        self.inst_subsample = inst_subsample
        self.inst_subsample_replacement = inst_subsample_replacement
        self.inst_splitter = inst_splitter
        assert model_subsample > 0
        assert (type(model_subsample) == int) or (model_subsample <= 1.)
        self.model_subsample = model_subsample
        self.model_subsample_replacement = model_subsample_replacement
        self.bagging_size = bagging_size
        self.init_top_k = init_top_k
        self.epsilon = epsilon
        self.multiprocessing = multiprocessing
        self.multiprocessing_num_cores = multiprocessing_num_cores
        self.enable_extreme = enable_extreme
        self.random_seed = random_seed
        logname = "ensemble_selection_%s.log"%time_utils._timestamp()
        self.logger = logging_utils._get_logger(config.LOG_DIR, logname)
        self.n_models = len(self.model_list)

    def _merge_pred(self, w1, p1, w2, p2):
        p = (w1 * p1 + w2 * p2) / (w1 + w2)
        # relevance is in [1,3]
        p = np.clip(p, 1., 3.)
        return p

    def _pick_random_models(self, sorted_models, seed):
        num_model = len(sorted_models)
        #### bagging for models
        rng = np.random.RandomState(seed)
        if type(self.model_subsample) == int:
            sample_size = min(num_model, self.model_subsample)
        else:
            sample_size = int(num_model * self.model_subsample)
        if self.model_subsample_replacement:
            index_base = rng.randint(num_model, size=sample_size)
        else:
            index_base = rng.permutation(num_model)[:sample_size]
        this_sorted_models = [sorted_models[i] for i in sorted(index_base)]
        return this_sorted_models

    def _ens_obj_generic(self, weight2, p1_list, weight1, p2_list, 
                        true_label_list, numBSTMatrix, bst_inst_idx):
        rmse_cv = np.zeros((config.N_RUNS, config.N_FOLDS), dtype=float)
        for run in range(config.N_RUNS):
            for fold in range(config.N_FOLDS):
                numBST = numBSTMatrix[run,fold]
                bidx = bst_inst_idx[run,fold,:numBST].tolist()
                p1 = p1_list[run,fold,bidx]
                p2 = p2_list[run,fold,bidx]
                true_label = true_label_list[run,fold,bidx]
                p_ens = self._merge_pred(weight1, p1, weight2, p2)
                rmse_cv[run,fold] = dist_utils._rmse(p_ens, true_label)
        rmse_mean = np.mean(rmse_cv)
        rmse_std = np.std(rmse_cv)
        return rmse_mean, rmse_std

    def _ens_obj_scipy(self, weight2, p1_list, weight1, p2_list, 
                        true_label_list, numBSTMatrix, bst_inst_idx):
        rmse_mean, rmse_std = self._ens_obj_generic(weight2, p1_list, weight1, p2_list, 
            true_label_list, numBSTMatrix, bst_inst_idx)
        return rmse_mean

    ## scipy.optimize.fmin for the best weight
    def _find_optim_weight_scipy(self, p_ens_list_valid_tmp, pred_list_valid, Y_list_valid, 
                                    numBSTMatrix, bst_inst_idx, w_ens, model_index_dict, model):
        this_p_list_valid = pred_list_valid[model_index_dict[model]]
        if w_ens == 0:
            this_w = 1.
            best_rmse_mean, best_rmse_std = self._ens_obj_generic(this_w, p_ens_list_valid_tmp, 
                0., this_p_list_valid, Y_list_valid, numBSTMatrix, bst_inst_idx)
        else:
            obj = lambda weight2: self._ens_obj_scipy(weight2, p_ens_list_valid_tmp, 
                1., this_p_list_valid, Y_list_valid, numBSTMatrix, bst_inst_idx)
            weight2_init = np.random.uniform(self.w_min, self.w_max, size=1)[0]
            xopt = scipy.optimize.fmin(obj, weight2_init, maxiter=self.weight_opt_max_evals, disp=False)
            this_w = xopt[0]
            best_rmse_mean, best_rmse_std = self._ens_obj_generic(this_w, p_ens_list_valid_tmp, 
                1., this_p_list_valid, Y_list_valid, numBSTMatrix, bst_inst_idx)
            this_w *= w_ens

        return best_rmse_mean, best_rmse_std, model, this_w

    def _ens_predict(self, best_bagged_model_list, best_bagged_model_weight):
        bagging_size = len(best_bagged_model_list)
        for bagging_iter in range(bagging_size):
            w_ens = 0.
            iter = 0
            for model,w in zip(best_bagged_model_list[bagging_iter], best_bagged_model_weight[bagging_iter]):
                iter += 1
                pred_file = "%s/All/test.pred.%s.csv" % (self.model_folder, model)
                df = pd.read_csv(pred_file, dtype=float)
                this_p_valid = df["prediction"].values
                this_p_valid = np.clip(this_p_valid, 1., 3.)
                if iter == 1:
                    p_ens_valid = np.zeros((this_p_valid.shape[0]),dtype=float)
                    id_test = df["id"].values.astype(int)
                p_ens_valid = self._merge_pred(w_ens, p_ens_valid, w, this_p_valid)
                w_ens += w
            
            if bagging_iter == 0:
                p_ens_score_bag = np.zeros((p_ens_valid.shape[0], bagging_size), dtype=float)

            p_ens_score_bag[:,bagging_iter] = p_ens_valid
        ##
        p_ens_score_bag = np.mean(p_ens_score_bag, axis=1)

        output = pd.DataFrame({"id": id_test, "relevance": p_ens_score_bag})
        return output

    def go(self):

        ## initialization
        pred_list_valid = np.zeros((self.n_models, config.N_RUNS, config.N_FOLDS, config.VALID_SIZE_MAX), dtype=float)
        Y_list_valid = np.zeros((config.N_RUNS, config.N_FOLDS, config.VALID_SIZE_MAX), dtype=float)
        numValidMatrix = np.zeros((config.N_RUNS, config.N_FOLDS), dtype=int)
        p_ens_list_valid = np.zeros((config.N_RUNS, config.N_FOLDS, config.VALID_SIZE_MAX), dtype=float)

        bst_inst_idx = np.zeros((config.N_RUNS, config.N_FOLDS, config.VALID_SIZE_MAX), dtype=float)
        numBSTMatrix = np.zeros((config.N_RUNS, config.N_FOLDS), dtype=int)
        oob_inst_idx = np.zeros((config.N_RUNS, config.N_FOLDS, config.VALID_SIZE_MAX), dtype=float)
        numOOBMatrix = np.zeros((config.N_RUNS, config.N_FOLDS), dtype=int)

        self.logger.info("Perform Extreme Ensemble Selection...")
        ## model index
        model_index_dict = dict(zip(self.model_list, range(self.n_models)))
        model_rmse_dict = dict(zip(self.model_list, [0]*self.n_models))
        self.logger.info("="*80)
        self.logger.info("Load model...")
        for model in self.model_list:
            self.logger.info("model: %s" % model)
            model_id = model_index_dict[model]
            rmse_cv = np.zeros((config.N_RUNS, config.N_FOLDS), dtype=float)
            ## load model
            for run in range(config.N_RUNS):
                for fold in range(config.N_FOLDS):
                    path = "%s/Run%d" % (self.model_folder, run+1)
                    pred_file = "%s/valid.pred.%s.csv" % (path, model)

                    this_p_valid = pd.read_csv(pred_file, dtype=float)
                    numValidMatrix[run,fold] = this_p_valid.shape[0]
                    numValid = numValidMatrix[run,fold]
                    this_target = this_p_valid["target"].values
                    this_p_valid = this_p_valid["prediction"].values
                    pred_list_valid[model_id,run,fold,:numValid] = np.clip(this_p_valid, 1., 3.)
                    Y_list_valid[run,fold,:numValid] = this_target

                    ##
                    rmse_cv[run,fold] = dist_utils._rmse(pred_list_valid[model_id,run,fold,:numValid], 
                                                            Y_list_valid[run,fold,:numValid])     

            self.logger.info("rmse: %.6f (%.6f)" % (np.mean(rmse_cv), np.std(rmse_cv)))
            model_rmse_dict[model] = (np.mean(rmse_cv), np.std(rmse_cv))
        self.logger.info("%d models in total." % self.n_models)

        sorted_models = sorted(model_rmse_dict.items(), key=lambda x: x[1][0])
            
        # greedy ensemble
        self.logger.info("="*80)
        best_bagged_model_list = [[]]*self.bagging_size
        best_bagged_model_weight = [[]]*self.bagging_size
        score_valid_bag_mean = np.nan * np.zeros((config.N_RUNS, config.N_FOLDS, config.VALID_SIZE_MAX, self.bagging_size), dtype=float)
        rmse_cv_mean_mean_lst = [0]*self.bagging_size
        rmse_cv_mean_std_lst = [0]*self.bagging_size
        for bagging_iter in range(self.bagging_size):
            seed_model = self.random_seed + 100 * bagging_iter
            if not self.enable_extreme:
                this_sorted_models = self._pick_random_models(sorted_models, seed_model)
            #### instance level subsampling
            for run in range(config.N_RUNS):
                for fold in range(config.N_FOLDS):
                    if self.inst_splitter is None:
                        # GENERAL APPROACH
                        seed_inst = self.random_seed + 1000 * bagging_iter + 100 * run + 10 * fold
                        rng_inst = np.random.RandomState(seed_inst)
                        numValid = numValidMatrix[run,fold]
                        if self.inst_subsample_replacement:
                            sss = StratifiedShuffleSplitReplacement(Y_list_valid[run,fold,:numValid], n_iter=1,
                                test_size=1.-self.inst_subsample, random_state=seed_inst)
                            iidx, oidx = list(sss)[0]
                        else:
                            if self.inst_subsample < 1:
                                # Stratified ShuffleSplit
                                sss = ShuffleSplit(len(Y_list_valid[run,fold,:numValid]), n_iter=1,
                                    test_size=1.-self.inst_subsample, random_state=seed_inst)
                                iidx, oidx = list(sss)[0]
                            elif self.inst_subsample == 1:
                                # set iidx (trianing) the same as oidx (validation)
                                iidx = np.arange(numValid)
                                oidx = np.arange(numValid)
                    else:
                        iidx, oidx = self.inst_splitter[run]
                    numBSTMatrix[run,fold] = len(iidx)
                    bst_inst_idx[run,fold,:numBSTMatrix[run,fold]] = iidx
                    numOOBMatrix[run,fold] = len(oidx)
                    oob_inst_idx[run,fold,:numOOBMatrix[run,fold]] = oidx

            #print this_model_list
            best_model_list = []
            best_model_weight = []
            best_model_rmse = []
            best_rmse = 0
            best_rmse_std = 0
            best_model = None
            p_ens_list_valid_tmp = np.zeros((config.N_RUNS, config.N_FOLDS, config.VALID_SIZE_MAX), dtype=float)
            #### Technique: Ensemble Initialization
            iter = 0
            w_ens, this_w = 0.0, 1.0
            if self.init_top_k > 0:
                # self.logger.info("** Ensemble Initialization **")
                # init_top_k = min(init_top_k, num_model)
                rmse_cv = np.zeros((config.N_RUNS, config.N_FOLDS), dtype=float)
                for cnt in range(self.init_top_k):
                    iter += 1
                    start = time.time()
                    seed_model = self.random_seed + 100 * bagging_iter + 10 * iter
                    if self.enable_extreme:
                        this_sorted_models = self._pick_random_models(sorted_models, seed_model)
                        best_model,(rmse,rmse_std) = this_sorted_models[0]
                    else:
                        best_model,(rmse,rmse_std) = this_sorted_models[cnt]
                    this_p_list_valid = pred_list_valid[model_index_dict[best_model]]
                    for run in range(config.N_RUNS):
                        for fold in range(config.N_FOLDS):
                            numValid = numValidMatrix[run,fold]
                            numBST = numBSTMatrix[run,fold]
                            bidx = bst_inst_idx[run,fold,:numBST].tolist()
                            p_ens_list_valid_tmp[run,fold,:numValid] = self._merge_pred(
                                w_ens, p_ens_list_valid_tmp[run,fold,:numValid], 
                                this_w, this_p_list_valid[run,fold,:numValid])
                            true_label = Y_list_valid[run,fold,bidx]
                            rmse_cv[run,fold] = dist_utils._rmse(p_ens_list_valid_tmp[run,fold,bidx], true_label)
                    end = time.time()
                    best_weight = this_w
                    best_rmse = np.mean(rmse_cv)
                    best_rmse_std = np.std(rmse_cv)

                    self.logger.info("Iter: %d (%.2fs)" % (iter, (end - start)))
                    self.logger.info("     model: %s" % best_model)
                    self.logger.info("     weight: %s" % best_weight)
                    self.logger.info("     rmse: %.6f (%.6f)" % (best_rmse, best_rmse_std))

                    best_model_list.append(best_model)
                    best_model_weight.append(best_weight)
                    w_ens += best_weight

            #### Technique: Ensemble Selection with Replacement
            while True:
                iter += 1
                seed_model = self.random_seed + 100 * bagging_iter + 10 * iter
                if self.enable_extreme:
                    this_sorted_models = self._pick_random_models(sorted_models, seed_model)
                if self.multiprocessing:
                    start = time.time()
                    models_tmp = [model for model,(_,_) in this_sorted_models]
                    best_trial_rmse_mean_lst, best_trial_rmse_std_lst, model_lst, this_w_lst = \
                        zip(*Parallel(n_jobs=self.multiprocessing_num_cores)(
                            delayed(self._find_optim_weight_scipy)(
                                p_ens_list_valid_tmp, pred_list_valid, Y_list_valid, numBSTMatrix, 
                                bst_inst_idx, w_ens, model_index_dict, m
                                ) for m in models_tmp
                            ))
                    ##
                    ind_best = np.argmin(best_trial_rmse_mean_lst)
                    best_trial_rmse_mean = best_trial_rmse_mean_lst[ind_best]
                    best_trial_rmse_std = best_trial_rmse_std_lst[ind_best]
                    model = model_lst[ind_best]
                    this_w = this_w_lst[ind_best]
                    if best_trial_rmse_mean < best_rmse:
                        best_rmse, best_rmse_std = best_trial_rmse_mean, best_trial_rmse_std
                        best_model, best_weight = model, this_w
                    end = time.time()
                else:
                    start = time.time()
                    for model,(_,_) in this_sorted_models:
                        best_trial_rmse_mean, best_trial_rmse_std, model, this_w = \
                            self._find_optim_weight_scipy(
                            p_ens_list_valid_tmp, pred_list_valid, Y_list_valid, numBSTMatrix, 
                            bst_inst_idx, w_ens, model_index_dict, model)
                        if best_trial_rmse_mean < best_rmse:
                            best_rmse, best_rmse_std = best_trial_rmse_mean, best_trial_rmse_std
                            best_model, best_weight = model, this_w
                    end = time.time()
                if best_model is None:
                    break
                if len(best_model_rmse) > 1 and (best_model_rmse[-1] - best_rmse < self.epsilon):
                    break

                ##
                self.logger.info("Iter: %d (%.2fs)" % (iter, (end - start)))
                self.logger.info("     model: %s" % best_model)
                self.logger.info("     weight: %s" % best_weight)
                self.logger.info("     rmse: %.6f (%.6f)" % (best_rmse, best_rmse_std))
                
                # valid
                this_p_list_valid = pred_list_valid[model_index_dict[best_model]]
                pred_raw_list = []
                true_label_list = []
                for run in range(config.N_RUNS):
                    for fold in range(config.N_FOLDS):
                        numValid = numValidMatrix[run,fold]
                        numBST = numBSTMatrix[run,fold]
                        bidx = bst_inst_idx[run,fold,:numBST].tolist()
                        p_ens_list_valid_tmp[run,fold,:numValid] = self._merge_pred(
                            w_ens, p_ens_list_valid_tmp[run,fold,:numValid], 
                            best_weight, this_p_list_valid[run,fold,:numValid])

                        pred_raw_list.append( p_ens_list_valid_tmp[run,fold,bidx] )
                        true_label_list.append( Y_list_valid[run,fold,bidx] )

                best_model_list.append(best_model)
                best_model_weight.append(best_weight)
                best_model_rmse.append(best_rmse)

                best_model = None
                w_ens += best_weight
            
            ## compute OOB score
            rmse_cv_mean = np.zeros((config.N_RUNS, config.N_FOLDS), dtype=float)
            for run in range(config.N_RUNS):
                for fold in range(config.N_FOLDS):
                    numValid = numValidMatrix[run,fold]
                    true_label = Y_list_valid[run,fold,:numValid]
                    numOOB = numOOBMatrix[run,fold]
                    oidx = oob_inst_idx[run,fold,:numOOB].tolist()                
                    pred_raw = p_ens_list_valid_tmp[run,fold,oidx]
                    ## mean
                    score_valid_bag_mean[run,fold,oidx,bagging_iter] = pred_raw
                    pred_mean = np_utils._array_mean(score_valid_bag_mean[run,fold,:numValid,:(bagging_iter+1)])
                    non_nan_idx = pred_mean != config.MISSING_VALUE_NUMERIC
                    rmse_cv_mean[run,fold] = dist_utils._rmse(pred_mean[non_nan_idx], true_label[non_nan_idx])
            self.logger.info("-"*80)
            self.logger.info( "Bag: %d"% (bagging_iter+1))
            self.logger.info( "rmse-mean: %.6f (%.6f)" % (np.mean(rmse_cv_mean), np.std(rmse_cv_mean)))
            self.logger.info("-"*80)

            best_bagged_model_list[bagging_iter] = best_model_list
            best_bagged_model_weight[bagging_iter] = best_model_weight

            ## save the current prediction
            mr = "R" + str(self.model_subsample_replacement).upper()[0]
            ir = "R" + str(self.inst_subsample_replacement).upper()[0]
            ## mean
            best_rmse_mean = np.mean(rmse_cv_mean)
            best_rmse_std = np.std(rmse_cv_mean)
            output = self._ens_predict(best_bagged_model_list[:(bagging_iter+1)], 
                best_bagged_model_weight[:(bagging_iter+1)])
            sub_file = "%s_[MS%.2f_%s]_[IS%.2f_%s]_[Top%d]_[Bag%d]_[Mean%.6f]_[Std%.6f].mean.csv" % (
                self.subm_prefix, self.model_subsample, mr, self.inst_subsample, ir, 
                self.init_top_k, bagging_iter+1, best_rmse_mean, best_rmse_std)
            output.to_csv(sub_file, index=False)
            rmse_cv_mean_mean_lst[bagging_iter] = best_rmse_mean
            rmse_cv_mean_std_lst[bagging_iter] = best_rmse_std

            ## plot OOB score
            x = np.arange(1,bagging_iter+2,1)
            label = "Mean (Best = %.6f, Bag = %d)"%(
                    np.min(rmse_cv_mean_mean_lst[:(bagging_iter+1)]), 
                    np.argmin(rmse_cv_mean_mean_lst[:(bagging_iter+1)])+1)
            plt.errorbar(x, rmse_cv_mean_mean_lst[:(bagging_iter+1)], 
                yerr=rmse_cv_mean_std_lst[:(bagging_iter+1)], 
                fmt='-o', label=label)
            plt.xlim(1, self.bagging_size)
            plt.title("Extreme Ensemble Selection RMSE")
            plt.xlabel("Bag")
            plt.ylabel("CV/OOB RMSE")
            plt.legend(loc="upper right")
            fig_file = "%s/ensemble_selection_%d.pdf"%(config.FIG_DIR, bagging_iter+1)
            plt.savefig(fig_file)
            plt.clf()


#--------------------------- Main ---------------------------
def main(options):

    # create sub folder
    subm_folder = "%s/ensemble_selection"%config.SUBM_DIR
    os_utils._create_dirs( [subm_folder] )
    subm_prefix = "%s/test.pred.[%s]" % (subm_folder, options.outfile)

    # get model list
    log_folder = "%s/level%d_models"%(config.LOG_DIR, options.level-1)
    model_list = get_model_list(log_folder, options.size)

    # get instance splitter
    if options.level not in [2, 3]:
        inst_splitter = None
    elif options.level == 2:
        inst_splitter = splitter_level2
    elif options.level == 3:
        inst_splitter = splitter_level3

    ees = ExtremeEnsembleSelection(
            model_folder=config.OUTPUT_DIR, 
            model_list=model_list, 
            subm_prefix=subm_prefix, 
            weight_opt_max_evals=options.weight_opt_max_evals, 
            w_min=-1., 
            w_max=1., 
            inst_subsample=options.inst_subsample,
            inst_subsample_replacement=options.inst_subsample_replacement,
            inst_splitter=inst_splitter,
            model_subsample=options.model_subsample,
            model_subsample_replacement=options.model_subsample_replacement,
            bagging_size=options.bagging_size, 
            init_top_k=options.init_top_k,
            epsilon=options.epsilon,
            multiprocessing=False, 
            multiprocessing_num_cores=config.NUM_CORES,
            enable_extreme=options.enable_extreme,
            random_seed=config.RANDOM_SEED
        )
    ees.go()


def parse_args(parser):
    parser.add_option("-l", "--level", type="int", dest="level", 
        default=2, help="level of base models")
    parser.add_option("-s", "--size", type="int", dest="size", 
        default=10, help="size of each model")
    parser.add_option("-o", "--outfile", type="string", dest="outfile",
        default="extreme_ensemble_selection", help="output model name")
    parser.add_option("-b", "--bag", type="int", dest="bagging_size",
        default=10, help="bagging_size")
    parser.add_option("-t", "--init", type="int", dest="init_top_k", 
        default=5, help="init_top_k")
    parser.add_option("-i", "--inst_subsample", type="float", dest="inst_subsample", 
        default=0.5, help="inst_subsample")
    parser.add_option("-I", "--inst_subsample_replacement", action="store_true", 
        dest="inst_subsample_replacement", 
        default=False, help="inst_subsample_replacement")
    parser.add_option("-m", "--model_subsample", type="float", dest="model_subsample", 
        default=1.0, help="model_subsample")
    parser.add_option("-M", "--model_subsample_replacement", action="store_true", 
        dest="model_subsample_replacement", 
        default=True, help="model_subsample_replacement")
    parser.add_option("-e", "--epsilon", type="float", dest="epsilon", 
        default=0.0001, help="epsilon")
    parser.add_option("-w", "--weight_opt_max_evals", type="int", dest="weight_opt_max_evals", 
        default=5, help="weight_opt_max_evals")
    parser.add_option("-x", action="store_true", dest="enable_extreme", 
        default=True, help="enable_extreme")
    (options, args) = parser.parse_args()
    return options, args


if __name__ == "__main__":

    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
