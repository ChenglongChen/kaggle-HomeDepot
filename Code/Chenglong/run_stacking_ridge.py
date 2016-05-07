# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: script for testing 2nd & 3rd level model with reg_skl_ridge

"""

import os
from optparse import OptionParser

from utils import time_utils


def parse_args(parser):
    parser.add_option("-l", "--level", default=2, 
        type="int", dest="level", help="level")
    parser.add_option("-d", "--dim", default=0, 
        type="int", dest="dim", help="LSA dim")
    parser.add_option("-t", "--top", default=10, 
        type="int", dest="top", help="top N")
    parser.add_option("-c", "--corr", default=1.0,
        type="float", dest="corr", help="corr")
    parser.add_option("-L", "--learner", default="reg_skl_ridge", 
        type="string", dest="learner", help="learner")
    parser.add_option("-o", default=False, action="store_true", dest="refit_once",
        help="stacking refit_once")
    (options, args) = parser.parse_args()
    return options, args

def main(options):
    now = time_utils._timestamp_pretty()

    meta_conf = "level%d_feature_conf_meta_linear_%s"%(options.level, now)
    stacking_conf = "level%d_feature_conf_%s"%(options.level, now)
    feat_name = "level%d_meta_linear_%s"%(options.level, now)

    # get meta feature conf for `level` models
    cmd = "python get_feature_conf_linear_stacking.py -d %d -o %s.py"%(
        options.dim, meta_conf)
    os.system(cmd)

    # NOTE: using predictions from `level-1` models to generate features 
    # for `level` models
    cmd = "python get_stacking_feature_conf.py -l %d -t %d -o %s.py"%(
        options.level-1, options.top, stacking_conf)
    os.system(cmd)

    # generate feature for `level` models
    cmd = "python feature_combiner.py -l %d -c %s -m %s -n %s -s .csv -t %f"%(
        options.level, stacking_conf, meta_conf, feat_name, options.corr)
    os.system(cmd)

    # train `level` models
    if options.refit_once:
        cmd = "python task.py -m stacking -f %s -l %s -e 100 -o"%(feat_name, options.learner)
    else:
        cmd = "python task.py -m stacking -f %s -l %s -e 100"%(feat_name, options.learner)
    os.system(cmd)

if __name__ == "__main__":
    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
