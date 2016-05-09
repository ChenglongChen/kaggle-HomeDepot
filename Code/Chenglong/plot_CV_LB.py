# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: plot CV RMSE vs LB RMSE

"""

import numpy as np
import matplotlib.pyplot as plt

import config


def main():
    rmse_cv = [
        # [0.527408,0.000768],
        # [0.482010,0.000752],
        [0.470570,0.000740],
        [0.470197,0.000558],
        [0.470167,0.000492],
        [0.468127,0.000749],
        [0.467613,0.000617],
        [0.467570,0.000509],
        [0.463124,0.000934],
        [0.462973,0.001178],
        [0.462632,0.001026],
        [0.461406,0.001050],
        [0.460582,0.001128],
        [0.458092,0.000782],
        [0.457421,0.000848],
        [0.455473,0.001008],
        [0.450111,0.000749],
        [0.447134,0.001033],
        [0.438318,0.000786],
    ]
    rmse_lb = [
        # [0.52770,0.52690],
        # [0.48067,0.48071],
        [0.46982,0.47028],
        [0.46968,0.46931],
        [0.46986,0.46981],
        [0.46864,0.46837],
        [0.46569,0.46544],
        [0.46653,0.46623],
        [0.46263,0.46181],
        [0.46251,0.46180],
        [0.46185,0.46147],
        [0.45944,0.45900],
        [0.45993,0.45958],
        [0.45909,0.45860],
        [0.45816,0.45725],
        [0.45640,0.45533],
        [0.44967,0.44902],
        [0.44577,0.44457],
        [0.43996,0.43811],
    ]


    rmse_cv = np.asarray(rmse_cv, dtype=float)
    rmse_lb = np.asarray(rmse_lb, dtype=float)

    N = rmse_cv.shape[0]
    x = np.arange(1,N+1,1)
    label = "CV"
    plt.errorbar(x, rmse_cv[:,0], 
        yerr=2*rmse_cv[:,1], 
        fmt='-o', label=label)
    plt.plot(x, rmse_lb[:,0])
    plt.plot(x, rmse_lb[:,1])
    plt.xlim(1, N)
    plt.title("CV RMSE vs LB RMSE")
    plt.xlabel("#Sub")
    plt.ylabel("RMSE")
    plt.legend(["CV (+- 2std)", "Public LB", "Private LB"], loc="upper right")
    fig_file = "%s/CV_LB_Chenglong.pdf"%config.FIG_DIR
    plt.savefig(fig_file)
    plt.clf()


if __name__ == "__main__":
    main()
