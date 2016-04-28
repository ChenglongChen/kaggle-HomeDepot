# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: feature transformer

"""

from collections import Counter

from sklearn.base import BaseEstimator


#### adopted from @Ben Hamner's Python Benchmark code
## https://www.kaggle.com/benhamner/crowdflower-search-relevance/python-benchmark
def identity(x):
    return x


class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return self.transformer(X)


class ColumnSelector(BaseEstimator):
    def __init__(self, columns=-1):
        # assert (type(columns) == int) or (type(columns) == list)
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        if len(X.shape) == 1:
            return X
        elif self.columns == -1:
            return X
        else:
            return X[:,self.columns]


# feature mapper for mapping rare categorical values to a special case
# example
# mapper = FeatureMapper(10, 0)
# dfTrain = mapper.fit_transform(dfTrain, "Medical_History_2")
# dfTest = mapper.transform(dfTest, "Medical_History_2")
class FeatureMapper:
    def __init__(self, threshold, rare_code):
        self.threshold = threshold
        self.rare_code = rare_code
        self.counter = Counter()
        self.mapper = {}

    def fit(self, X):
        self.counter = Counter(X)
        if self.rare_code is None:
            most_freq = sorted(self.counter.items(),
                                key=lambda x: x[1],
                                reverse=True)[0][0]
            self.rare_code = most_freq
        self.mapper = {}
        for k,v in self.counter.items():
            if v < self.threshold:
                self.mapper[k] = self.rare_code
        return self

    def transform(self, X):
        Y = map(lambda x:self.mapper.get(x, x), X)
        return Y

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class CountFeaturizer:
    def __init__(self):
        self.mapper = Counter()

    def fit(self, X):
        self.mapper = Counter(X)
        s = sum(self.mapper.values())
        for k,v in self.mapper.items():
            self.mapper[k] = float(v) / s
        return self

    def transform(self, X):
        Y = map(lambda x:self.mapper.get(x, 0), X)
        return Y

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
