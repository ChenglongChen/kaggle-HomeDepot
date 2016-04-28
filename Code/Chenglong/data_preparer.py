# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: generate raw dataframe data

"""

import gc

import numpy as np
import pandas as pd

import config
from utils import pkl_utils


def main():
    # load provided data
    dfTrain = pd.read_csv(config.TRAIN_DATA, encoding="ISO-8859-1")
    dfTest = pd.read_csv(config.TEST_DATA, encoding="ISO-8859-1")
    dfAttr = pd.read_csv(config.ATTR_DATA)
    dfDesc = pd.read_csv(config.DESC_DATA)

    # 
    print("Train Mean: %.6f"%np.mean(dfTrain["relevance"]))
    print("Train Var: %.6f"%np.var(dfTrain["relevance"]))

    #
    dfTest["relevance"] = np.zeros((config.TEST_SIZE))
    dfAttr.dropna(how="all", inplace=True)
    dfAttr["value"] = dfAttr["value"].astype(str)

    # concat train and test
    dfAll = pd.concat((dfTrain, dfTest), ignore_index=True)
    del dfTrain
    del dfTest
    gc.collect()

    # merge product description
    dfAll = pd.merge(dfAll, dfDesc, on="product_uid", how="left")
    dfAll.fillna(config.MISSING_VALUE_STRING, inplace=True)
    del dfDesc
    gc.collect()

    # merge product brand
    dfBrand = dfAttr[dfAttr.name=="MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "product_brand"})
    dfAll = pd.merge(dfAll, dfBrand, on="product_uid", how="left")
    dfBrand["product_brand"] = dfBrand["product_brand"].values.astype(str)
    dfAll.fillna(config.MISSING_VALUE_STRING, inplace=True)
    del dfBrand
    gc.collect()

    # merge product color
    color_columns = ["product_color", "Color Family", "Color/Finish", "Color/Finish Family"]
    dfColor = dfAttr[dfAttr.name.isin(color_columns)][["product_uid", "value"]].rename(columns={"value": "product_color"})
    dfColor.dropna(how="all", inplace=True)
    _agg_color = lambda df: " ".join(list(set(df["product_color"])))
    dfColor = dfColor.groupby("product_uid").apply(_agg_color)
    dfColor = dfColor.reset_index(name="product_color")
    dfColor["product_color"] = dfColor["product_color"].values.astype(str)
    dfAll = pd.merge(dfAll, dfColor, on="product_uid", how="left")
    dfAll.fillna(config.MISSING_VALUE_STRING, inplace=True)
    del dfColor
    gc.collect()

    # merge product attribute
    _agg_attr = lambda df: config.ATTR_SEPARATOR.join(df["name"] + config.ATTR_SEPARATOR + df["value"])
    dfAttr = dfAttr.groupby("product_uid").apply(_agg_attr)
    dfAttr = dfAttr.reset_index(name="product_attribute_concat")
    dfAll = pd.merge(dfAll, dfAttr, on="product_uid", how="left")
    dfAll.fillna(config.MISSING_VALUE_STRING, inplace=True)
    del dfAttr
    gc.collect()
    
    # save data
    if config.TASK == "sample":
        dfAll = dfAll.iloc[:config.SAMPLE_SIZE].copy()
    pkl_utils._save(config.ALL_DATA_RAW, dfAll)

    # info
    dfInfo = dfAll[["id","relevance"]].copy()
    pkl_utils._save(config.INFO_DATA, dfInfo)


if __name__ == "__main__":
    main()
