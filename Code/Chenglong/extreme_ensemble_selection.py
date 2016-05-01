# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: ensemble selection module

"""

import csv
import sys
import time

import scipy
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from sklearn.utils.fixes import bincount
from sklearn.cross_validation import BaseShuffleSplit, ShuffleSplit, StratifiedShuffleSplit, check_random_state
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval

import config
from config import LOG_DIR
from utils import dist_utils, np_utils, logging_utils, time_utils
import matplotlib.pyplot as plt


def _merge_pred(w1, p1, w2, p2):
    p = (w1 * p1 + w2 * p2) / (w1 + w2)
    # relevance is in [1,3]
    p = np.clip(p, 1., 3.)
    return p


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


########################
## Ensemble Selection ##
########################
def _ens_predict(model_folder,
                best_bagged_model_list,
                best_bagged_model_weight):
    bagging_size = len(best_bagged_model_list)
    for bagging_iter in range(bagging_size):
        w_ens = 0
        iter = 0
        for model,w in zip(best_bagged_model_list[bagging_iter], best_bagged_model_weight[bagging_iter]):
            iter += 1
            pred_file = "%s/All/test.pred.%s.csv" % (model_folder, model)
            df = pd.read_csv(pred_file, dtype=float)
            this_p_valid = df["prediction"].values
            this_p_valid = np.clip(this_p_valid, 1., 3.)
            this_w = w
            if iter == 1:
                p_ens_valid = np.zeros((this_p_valid.shape[0]),dtype=float)
                id_test = df["id"].values.astype(int)
            p_ens_valid = _merge_pred(w_ens, p_ens_valid, this_w, this_p_valid)
            w_ens += this_w
        
        if bagging_iter == 0:
            p_ens_score_bag = np.zeros((p_ens_valid.shape[0], bagging_size), dtype=float)

        p_ens_score_bag[:,bagging_iter] = p_ens_valid
    ##
    p_ens_score_bag = np.mean(p_ens_score_bag, axis=1)

    d = {"id": id_test, "relevance": p_ens_score_bag}
    output = pd.DataFrame(d)
    return output


def _ens_obj_generic(weight2, p1_list, weight1, p2_list, true_label_list, numBSTMatrix, bootstrapping_inst_idx):
    rmse_cv = np.zeros((config.N_RUNS, config.N_FOLDS), dtype=float)
    for run in range(config.N_RUNS):
        for fold in range(config.N_FOLDS):
            numBST = numBSTMatrix[run,fold]
            bidx = bootstrapping_inst_idx[run,fold,:numBST].tolist()
            p1 = p1_list[run,fold,bidx]
            p2 = p2_list[run,fold,bidx]
            true_label = true_label_list[run,fold,bidx]
            p_ens = _merge_pred(weight1, p1, weight2, p2)
            rmse_cv[run,fold] = dist_utils._rmse(p_ens, true_label)
    rmse_mean = np.mean(rmse_cv)
    rmse_std = np.std(rmse_cv)
    return rmse_mean, rmse_std


def _ens_obj_hyperopt(param, p1_list, weight1, p2_list, true_label_list, numBSTMatrix, bootstrapping_inst_idx):
    weight2 = param['weight2']
    rmse_mean, rmse_std = _ens_obj_generic(weight2, p1_list, weight1, p2_list, true_label_list, numBSTMatrix, bootstrapping_inst_idx)
    return {'loss': rmse_mean, 'attachments': {'std': rmse_std}, 'status': STATUS_OK}


def _ens_obj_scipy(param, p1_list, weight1, p2_list, true_label_list, numBSTMatrix, bootstrapping_inst_idx):
    weight2 = param
    rmse_mean, rmse_std = _ens_obj_generic(weight2, p1_list, weight1, p2_list, true_label_list, numBSTMatrix, bootstrapping_inst_idx)
    # print rmse_mean
    return rmse_mean


#---------------------------------------------------------------------------------------------------------------------#
# Implement ensemble selection with the following techniques described in [1][2]:
# 1. Ensemble Initialization
# 2. Bagged Ensemble Selection
# 3. Ensemble Selection with Replacement
# 4. Cross-Validated Ensemble Selection
#
# Reference
# [1] Ensemble selection from libraries of models, R. Caruana, A. Niculescu-Mizil, G. Crew, and A. Ksikes.
# [2] Getting the Most Out of Ensemble Selection, Rich Caruana, Art Munson, and Alexandru Niculescu-Mizil
#---------------------------------------------------------------------------------------------------------------------#
def _find_optim_weight_hyperopt(p_ens_list_valid_tmp, pred_list_valid, Y_list_valid, numBSTMatrix, bootstrapping_inst_idx, weight_opt_max_evals, w_min, w_max, w_ens, model2idx, model):
    this_p_list_valid = pred_list_valid[model2idx[model]]

    ## hyperopt for the best weight
    trials = Trials()
    if w_ens == 0:
        param_space = {'weight2': 1.}
        obj = lambda param: _ens_obj_hyperopt(param, p_ens_list_valid_tmp, 0., this_p_list_valid, Y_list_valid, numBSTMatrix, bootstrapping_inst_idx)
    else:
        param_space = {'weight2': hp.uniform('weight2', w_min, w_max)}
        obj = lambda param: _ens_obj_hyperopt(param, p_ens_list_valid_tmp, 1., this_p_list_valid, Y_list_valid, numBSTMatrix, bootstrapping_inst_idx)
    best = fmin(obj, param_space, algo=tpe.suggest,
                trials=trials, max_evals=weight_opt_max_evals)
    best_params = space_eval(param_space, best)
    this_w = best_params['weight2']
    if w_ens != 0:
        this_w *= w_ens
    trial_rmses = np.asarray(trials.losses(), dtype=float)
    best_ind = np.argmin(trial_rmses)
    best_rmse_mean = trial_rmses[best_ind]
    best_rmse_std = trials.trial_attachments(trials.trials[best_ind])['std']
    return best_rmse_mean, best_rmse_std, model, this_w


## scipy.optimize.fmin for the best weight
def _find_optim_weight_scipy(p_ens_list_valid_tmp, pred_list_valid, Y_list_valid, numBSTMatrix, bootstrapping_inst_idx, weight_opt_max_evals, w_min, w_max, w_ens, model2idx, model):
    this_p_list_valid = pred_list_valid[model2idx[model]]
    if w_ens == 0:
        this_w = 1.
        best_rmse_mean, best_rmse_std = _ens_obj_generic(this_w, p_ens_list_valid_tmp, 0., this_p_list_valid, Y_list_valid, numBSTMatrix, bootstrapping_inst_idx)
    else:
        obj = lambda param: _ens_obj_scipy(param, p_ens_list_valid_tmp, 1., this_p_list_valid, Y_list_valid, numBSTMatrix, bootstrapping_inst_idx)
        weight2_init = np.random.uniform(w_min, w_max, size=1)[0]
        xopt = scipy.optimize.fmin(obj, weight2_init, maxiter=weight_opt_max_evals, disp=False)
        # print xopt
        this_w = xopt[0]
        best_rmse_mean, best_rmse_std = _ens_obj_generic(this_w, p_ens_list_valid_tmp, 1., this_p_list_valid, Y_list_valid, numBSTMatrix, bootstrapping_inst_idx)
        this_w *= w_ens

    return best_rmse_mean, best_rmse_std, model, this_w


def _pick_random_models(sorted_models, model_subsample, model_subsample_replacement, seed):
    num_model = len(sorted_models)
    #### bagging for models
    rng = np.random.RandomState(seed)
    if type(model_subsample) == int:
        sampleSize = min(num_model, model_subsample)
    else:
        sampleSize = int(num_model*model_subsample)
    if model_subsample_replacement:
        index_base = rng.randint(num_model, size=sampleSize)
    else:
        index_base = rng.permutation(num_model)[:sampleSize]
    this_sorted_models = [sorted_models[i] for i in sorted(index_base)]
    return this_sorted_models


def ensemble_selection(
    model_folder, 
    model_list, 
    subm_prefix,
    weight_opt_max_evals=10, 
    w_min=-1., 
    w_max=1.,
    inst_subsample=1.,
    inst_subsample_replacement=True,
    model_subsample=0.3,
    model_subsample_replacement=False,
    bagging_size=10, 
    init_top_k=5, 
    epsilon=0, 
    multiprocessing=False, 
    multiprocessing_num_cores=1,
    enable_extreme=False):

    logname = "ens_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(LOG_DIR, logname)

    assert inst_subsample > 0 and inst_subsample <= 1.
    assert model_subsample > 0
    assert (type(model_subsample) == int) or (model_subsample <= 1.)

    model_list = model_list

    ## load all the prediction
    pred_list_valid = np.zeros((len(model_list), config.N_RUNS, config.N_FOLDS, config.VALID_SIZE_MAX), dtype=float)
    Y_list_valid = np.zeros((config.N_RUNS, config.N_FOLDS, config.VALID_SIZE_MAX), dtype=float)
    numValidMatrix = np.zeros((config.N_RUNS, config.N_FOLDS), dtype=int)
    p_ens_list_valid = np.zeros((config.N_RUNS, config.N_FOLDS, config.VALID_SIZE_MAX), dtype=float)

    bootstrapping_inst_idx = np.zeros((config.N_RUNS, config.N_FOLDS, config.VALID_SIZE_MAX), dtype=float)
    numBSTMatrix = np.zeros((config.N_RUNS, config.N_FOLDS), dtype=int)
    oob_inst_idx = np.zeros((config.N_RUNS, config.N_FOLDS, config.VALID_SIZE_MAX), dtype=float)
    numOOBMatrix = np.zeros((config.N_RUNS, config.N_FOLDS), dtype=int)

    numTest = config.TEST_SIZE

    ## model to idx
    model2idx = dict()
    rmse_list = dict()
    for i,model in enumerate(model_list): 
        model2idx[model] = i
        rmse_list[model] = 0
    logger.info("="*50)
    logger.info("Load model...")
    for model in model_list:
        logger.info("model: %s" % model)
        model_id = model2idx[model]
        rmse_cv = np.zeros((config.N_RUNS, config.N_FOLDS), dtype=float)
        ## load cvf
        for run in range(config.N_RUNS):
            for fold in range(config.N_FOLDS):
                path = "%s/Run%d" % (model_folder, run+1)
                pred_file = "%s/valid.pred.%s.csv" % (path, model)

                this_p_valid = pd.read_csv(pred_file, dtype=float)
                numValidMatrix[run,fold] = this_p_valid.shape[0]
                numValid = numValidMatrix[run,fold]
                this_target = this_p_valid["target"].values
                this_p_valid = this_p_valid["prediction"].values
                pred_list_valid[model_id,run,fold,:numValid] = np.clip(this_p_valid, 1., 3.)
                Y_list_valid[run,fold,:numValid] = this_target

                ##
                rmse_cv[run,fold] = dist_utils._rmse(pred_list_valid[model_id,run,fold,:numValid], Y_list_valid[run,fold,:numValid])     

        logger.info("rmse: %.6f (%.6f)" % (np.mean(rmse_cv), np.std(rmse_cv)))
        rmse_list[model] = (np.mean(rmse_cv), np.std(rmse_cv))

    sorted_models = sorted(rmse_list.items(), key=lambda x: x[1][0])
        
    # greedy ensemble
    logger.info("="*50)
    #logger.info("Perform ensemble selection...")
    best_bagged_model_list = [[]]*bagging_size
    best_bagged_model_weight = [[]]*bagging_size
    score_valid_bag_mean = np.nan * np.zeros((config.N_RUNS, config.N_FOLDS, config.VALID_SIZE_MAX, bagging_size), dtype=float)
    num_model = len(model_list)
    rmse_cv_mean_mean_lst = [0]*bagging_size
    rmse_cv_mean_std_lst = [0]*bagging_size

    header = "Bag,Mean_rmse_mean,Mean_rmse_std\n"
    logger.info(header)
    logger.info("%d models in total." % num_model)
    logger.info("** Bagged Ensemble Selection **")
    for bagging_iter in range(bagging_size):
        seed_model = 2016 + 100 * bagging_iter
        if not enable_extreme:
            this_sorted_models = _pick_random_models(sorted_models, model_subsample, model_subsample_replacement, seed_model)
        #### instance level subsampling
        for run in range(config.N_RUNS):
            for fold in range(config.N_FOLDS):
                seed_inst = 2016 + 1000 * bagging_iter + 100 * run + 10 * fold
                rng_inst = np.random.RandomState(seed_inst)
                numValid = numValidMatrix[run,fold]
                if inst_subsample_replacement:
                    sss = StratifiedShuffleSplitReplacement(Y_list_valid[run,fold,:numValid], n_iter=1,
                        test_size=1.-inst_subsample, random_state=seed_inst)
                    iidx, oidx = list(sss)[0]
                else:
                    if inst_subsample < 1:
                        # Stratified ShuffleSplit
                        sss = ShuffleSplit(len(Y_list_valid[run,fold,:numValid]), n_iter=1,
                            test_size=1.-inst_subsample, random_state=seed_inst)
                        iidx, oidx = list(sss)[0]
                    elif inst_subsample == 1:
                        # set iidx (trianing) the same as oidx (validation)
                        iidx = np.arange(numValid)
                        oidx = np.arange(numValid)
                numBSTMatrix[run,fold] = len(iidx)
                bootstrapping_inst_idx[run,fold,:numBSTMatrix[run,fold]] = iidx
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
        if init_top_k > 0:
            # logger.info("** Ensemble Initialization **")
            # init_top_k = min(init_top_k, num_model)
            rmse_cv = np.zeros((config.N_RUNS, config.N_FOLDS), dtype=float)
            for cnt in range(init_top_k):
                iter += 1
                start = time.time()
                seed_model = 2016 + 100 * bagging_iter + 10 * iter
                if enable_extreme:
                    this_sorted_models = _pick_random_models(sorted_models, model_subsample, model_subsample_replacement, seed_model)
                    best_model,(rmse,rmse_std) = this_sorted_models[0]
                else:
                    best_model,(rmse,rmse_std) = this_sorted_models[cnt]
                this_p_list_valid = pred_list_valid[model2idx[best_model]]
                for run in range(config.N_RUNS):
                    for fold in range(config.N_FOLDS):
                        numValid = numValidMatrix[run,fold]
                        numBST = numBSTMatrix[run,fold]
                        bidx = bootstrapping_inst_idx[run,fold,:numBST].tolist()
                        p_ens_list_valid_tmp[run,fold,:numValid] = _merge_pred(w_ens, p_ens_list_valid_tmp[run,fold,:numValid], this_w, this_p_list_valid[run,fold,:numValid])
                        true_label = Y_list_valid[run,fold,bidx]
                        rmse_cv[run,fold] = dist_utils._rmse(p_ens_list_valid_tmp[run,fold,bidx], true_label)
                end = time.time()
                best_weight =  this_w
                best_rmse = np.mean(rmse_cv)
                best_rmse_std = np.std(rmse_cv)

                logger.info("Iter: %d (%.2fs)" % (iter, (end - start)))
                logger.info("     model: %s" % best_model)
                logger.info("     weight: %s" % best_weight)
                logger.info("     rmse: %.6f (%.6f)" % (best_rmse, best_rmse_std))

                best_model_list.append(best_model)
                best_model_weight.append(best_weight)
                w_ens += best_weight

        #### Technique: Ensemble Selection with Replacement
        # logger.info("** Ensemble Selection with Replacement **")
        # iter = 0
        while True:
            iter += 1
            seed_model = 2016 + 100 * bagging_iter + 10 * iter
            if enable_extreme:
                this_sorted_models = _pick_random_models(sorted_models, model_subsample, model_subsample_replacement, seed_model)
            if multiprocessing:
                start = time.time()
                models_tmp = [model for model,(_,_) in this_sorted_models]
                best_trial_rmse_mean_lst, best_trial_rmse_std_lst, model_lst, this_w_lst = \
                zip(*Parallel(n_jobs=multiprocessing_num_cores)(delayed(_find_optim_weight_scipy)(p_ens_list_valid_tmp, pred_list_valid, Y_list_valid, numBSTMatrix, bootstrapping_inst_idx, weight_opt_max_evals, w_min, w_max, w_ens, model2idx, m) for m in models_tmp))
                ##
                ind_best = np.argmin(best_trial_rmse_mean_lst)
                best_trial_rmse_mean = best_trial_rmse_mean_lst[ind_best]
                best_trial_rmse_std = best_trial_rmse_std_lst[ind_best]
                model = model_lst[ind_best]
                this_w = this_w_lst[ind_best]
                if best_trial_rmse_mean < best_rmse:
                    best_rmse, best_rmse_std, best_model, best_weight = best_trial_rmse_mean, best_trial_rmse_std, model, this_w
                end = time.time()
            else:
                start = time.time()
                for model,(_,_) in this_sorted_models:
                    best_trial_rmse_mean, best_trial_rmse_std, model, this_w = \
                    _find_optim_weight_scipy(p_ens_list_valid_tmp, pred_list_valid, Y_list_valid, numBSTMatrix, bootstrapping_inst_idx, weight_opt_max_evals, w_min, w_max, w_ens, model2idx, model)
                    if best_trial_rmse_mean < best_rmse:
                        best_rmse, best_rmse_std, best_model, best_weight = best_trial_rmse_mean, best_trial_rmse_std, model, this_w
                end = time.time()
            if (best_model is None) or (len(best_model_rmse) > 1 and  (best_model_rmse[-1] - best_rmse < epsilon)):
                break

            ##
            logger.info("Iter: %d (%.2fs)" % (iter, (end - start)))
            logger.info("     model: %s" % best_model)
            logger.info("     weight: %s" % best_weight)
            logger.info("     rmse: %.6f (%.6f)" % (best_rmse, best_rmse_std))
            
            # valid
            this_p_list_valid = pred_list_valid[model2idx[best_model]]
            pred_raw_list = []
            true_label_list = []
            for run in range(config.N_RUNS):
                for fold in range(config.N_FOLDS):
                    numValid = numValidMatrix[run,fold]
                    numBST = numBSTMatrix[run,fold]
                    bidx = bootstrapping_inst_idx[run,fold,:numBST].tolist()
                    p_ens_list_valid_tmp[run,fold,:numValid] = _merge_pred(w_ens, p_ens_list_valid_tmp[run,fold,:numValid], best_weight, this_p_list_valid[run,fold,:numValid])

                    pred_raw_list.append( p_ens_list_valid_tmp[run,fold,bidx] )
                    true_label_list.append( Y_list_valid[run,fold,bidx] )

            best_model_list.append(best_model)
            best_model_weight.append(best_weight)
            best_model_rmse.append(best_rmse)

            best_model = None
            w_ens += best_weight
        
        ## compute oob score
        ## if we are optimizing cdf on oob, this will be over estimate
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
                pred_mean = np_utils.arrayMean(score_valid_bag_mean[run,fold,:numValid,:(bagging_iter+1)])
                non_nan_idx = pred_mean != config.MISSING_VALUE_NUMERIC
                rmse_cv_mean[run,fold] = dist_utils._rmse(pred_mean[non_nan_idx], true_label[non_nan_idx])
        logger.info("="*20)
        logger.info( "\nBag %d"% (bagging_iter+1))
        logger.info( "rmse-mean: %.6f (%.6f)" % (np.mean(rmse_cv_mean), np.std(rmse_cv_mean)))
        logger.info("="*20)

        best_bagged_model_list[bagging_iter] = best_model_list
        best_bagged_model_weight[bagging_iter] = best_model_weight

        ## save the current prediction
        mr = "R" + str(model_subsample_replacement).upper()[0]
        ir = "R" + str(inst_subsample_replacement).upper()[0]
        ## mean
        best_rmse_mean = np.mean(rmse_cv_mean)
        best_rmse_std = np.std(rmse_cv_mean)
        output = _ens_predict(model_folder, best_bagged_model_list[:(bagging_iter+1)], best_bagged_model_weight[:(bagging_iter+1)])
        sub_file = "%s_[MS%.2f_%s]_[IS%.2f_%s]_[Top%d]_[Bag%d]_[Mean%.6f]_[Std%.6f].mean.csv" % (subm_prefix, model_subsample, mr, inst_subsample, ir, init_top_k, bagging_iter+1, best_rmse_mean, best_rmse_std)
        output.to_csv(sub_file, index=False)
        rmse_cv_mean_mean_lst[bagging_iter] = best_rmse_mean
        rmse_cv_mean_std_lst[bagging_iter] = best_rmse_std

        ## plot OOB QWK
        if config.PLATFORM == "Windows":
            x = np.arange(1,bagging_iter+2,1)
            plt.errorbar(x, rmse_cv_mean_mean_lst[:(bagging_iter+1)], 
                yerr=rmse_cv_mean_std_lst[:(bagging_iter+1)], 
                fmt='-o', label="Mean")
            plt.xlim(1, bagging_size)
            plt.xlabel("Bag")
            plt.ylabel("OOB RMSE")
            plt.legend(loc="upper right")
            fig_file = "/ens%d.pdf"%(bagging_iter+1)
            plt.savefig(config.FIG_DIR + fig_file)
            plt.clf()

        ## write to log
        to_write = "%d,%.5f,%.5f\n" % (
            bagging_iter+1, 
            rmse_cv_mean_mean_lst[bagging_iter],
            rmse_cv_mean_std_lst[bagging_iter]
        )
        logger.info(to_write)
