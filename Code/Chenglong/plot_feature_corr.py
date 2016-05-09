# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: plot correlation with target relevance for each feature group

"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt

import config


def is_feat_log(fname):
    pat = re.compile("generate_feature_(.+)_2016")
    groups = re.findall(pat, fname)
    if len(groups) > 0 and groups[0] != "group_relevance":
        return groups[0]
    return None


def grap_feat_line_corr(line):
    pat = re.compile("corr = (.+)")
    groups = re.findall(pat, line)
    if len(groups) > 0:
        return float(groups[0])
    return None


def grap_feat_line_name(line):
    pat = re.compile("INFO: (.+) \(\d+D\):")
    groups = re.findall(pat, line)
    if len(groups) > 0:
        return groups[0]
    return None    


def grap_feat_corr_dict(fname):
    d = {}
    with open("%s/feature/%s"%(config.LOG_DIR, fname), "r") as f:
        for line in f:
            corr = grap_feat_line_corr(line)
            if corr is not None:
                name = grap_feat_line_name(line)
                d[name] = (corr)
    return d.values()

def grap_all_feat_corr_dict():
    d = {}
    for fname in sorted(os.listdir("%s/feature"%(config.LOG_DIR))):
        name = is_feat_log(fname)
        if name is not None:
            d[name] = grap_feat_corr_dict(fname)
    return d

def main():
    rng = np.random.RandomState(config.RANDOM_SEED)
    sample_size = 50
    colors = 'rgbcmyk'
    d = grap_all_feat_corr_dict()
    keys = sorted(d.keys())
    N = len(keys)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for e,k in enumerate(keys, start=1):
        vals = d[k]
        # sample a few
        if len(vals) > sample_size:
            idx = rng.randint(len(vals), size=sample_size)
            vals = [vals[i] for i in idx]
        plt.bar(np.linspace(e-0.48,e+0.48,len(vals)), sorted(vals), 
            width=1./(len(vals)+5), color=colors[e % len(colors)])
    plt.xlabel("Feature Group", fontsize=13)
    plt.ylabel("Correlation Coef", fontsize=13)
    plt.xticks(range(1,N+1), fontsize=13)
    plt.yticks([-0.4, -0.2, 0, 0.2, 0.4], fontsize=13)
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_xlim([0, N+1])
    ax.set_ylim([-0.4, 0.4])
    pos1 = ax.get_position()
    pos2 = [pos1.x0 - 0.075, pos1.y0 + 0.15,  pos1.width * 1.2, pos1.height * 0.9] 
    ax.set_position(pos2)
    plt.show()


if __name__ == "__main__":
    main()
