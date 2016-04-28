# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: word2vec & doc2vec trainer

"""

import os
import sys

import pandas as pd
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import LabeledSentence

import config
from utils import nlp_utils
from utils import logging_utils, pkl_utils, time_utils


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


#---------------------- Word2Vec ----------------------
class DataFrameSentences(object):
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns

    def __iter__(self):
        for column in self.columns:
            for sentence in self.df[column]:
                tokens = nlp_utils._tokenize(sentence, token_pattern)
                yield tokens


class DataFrameWord2Vec:
    def __init__(self, df, columns, model_param):
        self.df = df
        self.columns = columns
        self.model_param = model_param
        self.model = Word2Vec(sg=self.model_param["sg"], 
                                hs=self.model_param["hs"], 
                                alpha=self.model_param["alpha"],
                                min_alpha=self.model_param["alpha"],
                                min_count=self.model_param["min_count"], 
                                size=self.model_param["size"], 
                                sample=self.model_param["sample"], 
                                window=self.model_param["window"], 
                                workers=self.model_param["workers"])

    def train(self):
        # build vocabulary
        self.sentences = DataFrameSentences(self.df, self.columns)
        self.model.build_vocab(self.sentences)
        # train for n_epoch
        for i in range(self.model_param["n_epoch"]):
            self.sentences = DataFrameSentences(self.df, self.columns)
            self.model.train(self.sentences)
            self.model.alpha *= self.model_param["learning_rate_decay"]
            self.model.min_alpha = self.model.alpha
        return self

    def save(self, model_dir, model_name):
        fname = os.path.join(model_dir, model_name)
        self.model.save(fname)


def train_word2vec_model(df, columns):
    model_param = {
        "alpha": config.EMBEDDING_ALPHA,
        "learning_rate_decay": config.EMBEDDING_LEARNING_RATE_DECAY,
        "n_epoch": config.EMBEDDING_N_EPOCH,
        "sg": 1,
        "hs": 1,
        "min_count": config.EMBEDDING_MIN_COUNT,
        "size": config.EMBEDDING_DIM,
        "sample": 0.001,
        "window": config.EMBEDDING_WINDOW,
        "workers": config.EMBEDDING_WORKERS,
    }
    model_dir = config.WORD2VEC_MODEL_DIR
    model_name = "Homedepot-word2vec-D%d-min_count%d.model"%(
                    model_param["size"], model_param["min_count"])

    word2vec = DataFrameWord2Vec(df, columns, model_param)
    word2vec.train()
    word2vec.save(model_dir, model_name)


#---------------------- Doc2Vec ----------------------
class DataFrameLabelSentences(object):
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns
        self.cnt = -1
        self.sent_label = {}

    def __iter__(self):
        for column in self.columns:
            for sentence in self.df[column]:
                if not sentence in self.sent_label:
                    self.cnt += 1
                    self.sent_label[sentence] = "SENT_%d"%self.cnt
                tokens = nlp_utils._tokenize(sentence, token_pattern)
                yield LabeledSentence(words=tokens, tags=[self.sent_label[sentence]])


class DataFrameDoc2Vec(DataFrameWord2Vec):
    def __init__(self, df, columns, model_param):
        super().__init__(df, columns, model_param)
        self.model = Doc2Vec(dm=self.model_param["dm"], 
                                hs=self.model_param["hs"], 
                                alpha=self.model_param["alpha"],
                                min_alpha=self.model_param["alpha"],
                                min_count=self.model_param["min_count"], 
                                size=self.model_param["size"], 
                                sample=self.model_param["sample"], 
                                window=self.model_param["window"], 
                                workers=self.model_param["workers"])
    def train(self):
        # build vocabulary
        self.sentences = DataFrameLabelSentences(self.df, self.columns)
        self.model.build_vocab(self.sentences)
        # train for n_epoch
        for i in range(self.model_param["n_epoch"]):
            self.sentences = DataFrameLabelSentences(self.df, self.columns)
            self.model.train(self.sentences)
            self.model.alpha *= self.model_param["learning_rate_decay"]
            self.model.min_alpha = self.model.alpha
        return self

    def save(self, model_dir, model_name):
        fname = os.path.join(model_dir, model_name)
        self.model.save(fname)
        pkl_utils._save("%s.sent_label"%fname, self.sentences.sent_label)


def train_doc2vec_model(df, columns):
    model_param = {
        "alpha": config.EMBEDDING_ALPHA,
        "learning_rate_decay": config.EMBEDDING_LEARNING_RATE_DECAY,
        "n_epoch": config.EMBEDDING_N_EPOCH,
        "sg": 1, # not use
        "dm": 1,
        "hs": 1,
        "min_count": config.EMBEDDING_MIN_COUNT,
        "size": config.EMBEDDING_DIM,
        "sample": 0.001,
        "window": config.EMBEDDING_WINDOW,
        "workers": config.EMBEDDING_WORKERS,
    }
    model_dir = config.DOC2VEC_MODEL_DIR
    model_name = "Homedepot-doc2vec-D%d-min_count%d.model"%(
                    model_param["size"], model_param["min_count"])

    doc2vec = DataFrameDoc2Vec(df, columns, model_param)
    doc2vec.train()
    doc2vec.save(model_dir, model_name)


#---------------------- Main ----------------------
if __name__ == "__main__":
    df = pkl_utils._load(config.ALL_DATA_LEMMATIZED)
    columns = ["search_term", "search_term_alt", "product_title", "product_description",
                "product_attribute", "product_brand", "product_color"]
    columns = [col for col in columns if col in df.columns]

    if len(sys.argv) >= 2:
        for w in sys.argv[1].split(","):
            if w == "word2vec":
                train_word2vec_model(df, columns)
            elif w == "doc2vec":
                train_doc2vec_model(df, columns)
            else:
                print("Skip: %s"%w)
                continue
    else:
        train_doc2vec_model(df, columns)
        train_word2vec_model(df, columns)
