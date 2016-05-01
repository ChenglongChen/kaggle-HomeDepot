# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: generate ensembled prediction via extreme ensemble selection

"""

from optparse import OptionParser

import config
from utils import os_utils
from get_stacking_feature_conf import get_model_list
from extreme_ensemble_selection import ensemble_selection


def main(options):

    # create sub folder
    if options.enable_extreme:
        subm_folder = "%s/extreme_ensemble_selection"%config.SUBM_DIR
    else:
        subm_folder = "%s/ensemble_selection"%config.SUBM_DIR
    os_utils._create_dirs( [subm_folder] )

    # get model list
    model_list = get_model_list(options.log_folder, options.size)
    
    subm_prefix = "%s/test.pred.[%s]" % (subm_folder, options.outfile)       
    ensemble_selection(
        model_folder=config.OUTPUT_DIR, 
        model_list=model_list, 
        subm_prefix=subm_prefix, 
        weight_opt_max_evals=10, 
        w_min=-1, 
        w_max=1, 
        inst_subsample=options.inst_subsample,
        inst_subsample_replacement=options.inst_subsample_replacement,
        model_subsample=options.model_subsample,
        model_subsample_replacement=options.model_subsample_replacement,
        bagging_size=options.bagging_size, 
        init_top_k=options.init_top_k,
        epsilon=options.epsilon,
        multiprocessing=False, 
        multiprocessing_num_cores=config.NUM_CORES,
        enable_extreme=options.enable_extreme
    )


def parse_args(parser):
    parser.add_option("-l", "--log_folder", type="string", dest="log_folder", 
        default="%s/level1_models"%config.LOG_DIR, help="log folder")
    parser.add_option("-s", "--size", type="int", dest="size", 
        default=5, help="size of each model")
    parser.add_option("-o", "--outfile", type="string", dest="outfile",
        default="extreme_ensemble_selection", help="output feature name")
    parser.add_option("-b", "--bag", type="int", dest="bagging_size",
        default=20, help="bagging_size")
    parser.add_option("-t", "--init", type="int", dest="init_top_k", 
        default=5, help="init_top_k")
    parser.add_option("-i", "--inst_subsample", type="float", dest="inst_subsample", 
        default=0.5, help="inst_subsample")
    parser.add_option("-I", "--inst_subsample_replacement", action="store_true", 
        dest="inst_subsample_replacement", default=False, 
        help="inst_subsample_replacement")
    parser.add_option("-m", "--model_subsample", type="float", dest="model_subsample", 
        default=0.5, help="model_subsample")
    parser.add_option("-M", "--model_subsample_replacement", action="store_true", 
        dest="model_subsample_replacement", default=True, 
        help="model_subsample_replacement")
    parser.add_option("-e", "--epsilon", type="float", dest="epsilon", 
        default=0.00001, help="epsilon")
    parser.add_option("-x", action="store_true", dest="enable_extreme", 
        default=True, help="enable_extreme")
    (options, args) = parser.parse_args()
    return options, args


if __name__ == "__main__":

    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
