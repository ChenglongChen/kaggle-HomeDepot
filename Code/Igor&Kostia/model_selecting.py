# -*- coding: utf-8 -*-
"""
Code for selecting top N models and build stacker on them.
Competition: HomeDepot Search Relevance
Author: Kostia Omelianchuk
Team: Turing test
"""

from config_IgorKostia import *

import os
import pandas as pd
import xgboost as xgb 
import csv
import random
import numpy as np
import scipy as sp
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR,LinearSVC
from sklearn import neighbors
from sklearn import linear_model
from time import time
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomTreesEmbedding
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from math import sqrt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn import preprocessing


drop_list= []


#loading models


#9 model
train_f_1000 = pd.read_csv(MODELS_DIR+'/train_first_1000.csv', encoding="utf-8")
train_s_1000 = pd.read_csv(MODELS_DIR+'/train_second_1000.csv', encoding="utf-8")
train_f_1001 = pd.read_csv(MODELS_DIR+'/train_first_1001.csv', encoding="utf-8")

train_f_2000 = pd.read_csv(MODELS_DIR+'/train_first_2000.csv', encoding="utf-8")
train_s_2000 = pd.read_csv(MODELS_DIR+'/train_second_2000.csv', encoding="utf-8")




test_f_1000 = pd.read_csv(MODELS_DIR+'/test_first_1000.csv', encoding="utf-8")
test_s_1000 = pd.read_csv(MODELS_DIR+'/test_second_1000.csv', encoding="utf-8")
test_f_1001 = pd.read_csv(MODELS_DIR+'/test_first_1001.csv', encoding="utf-8")

test_f_2000 = pd.read_csv(MODELS_DIR+'/test_first_2000.csv', encoding="utf-8")
test_s_2000 = pd.read_csv(MODELS_DIR+'/test_second_2000.csv', encoding="utf-8")

#6 model
train_f_3000 = pd.read_csv(MODELS_DIR+'/train_first_3000.csv', encoding="utf-8")
train_s_3000 = pd.read_csv(MODELS_DIR+'/train_second_3000.csv', encoding="utf-8")

test_f_3000 = pd.read_csv(MODELS_DIR+'/test_first_3000.csv', encoding="utf-8")
test_s_3000 = pd.read_csv(MODELS_DIR+'/test_second_3000.csv', encoding="utf-8")

#6 model only kostia features
train_f_3010 = pd.read_csv(MODELS_DIR+'/train_first_3010.csv', encoding="utf-8")
test_f_3010 = pd.read_csv(MODELS_DIR+'/test_first_3010.csv', encoding="utf-8")

#6 model (4SVR + 2xgb) on corelated fetures
train_f_3020 = pd.read_csv(MODELS_DIR+'/train_first_3020.csv', encoding="utf-8")
test_f_3020 = pd.read_csv(MODELS_DIR+'/test_first_3020.csv', encoding="utf-8")



train=pd.DataFrame()
test=pd.DataFrame()

train = pd.concat([train_f_1000, train_s_1000, train_f_1001, train_f_2000, train_s_2000, train_f_3000, train_s_3000, train_f_3010,train_f_3020], axis=1)
test = pd.concat([test_f_1000, test_s_1000, test_f_1001, test_f_2000, test_s_2000, test_f_3000, test_s_3000 , test_f_3010, test_f_3020], axis=1)


#adding_some_metafeatures
df_all = pd.read_csv(FEATURES_DIR+'/df_basic_features.csv', encoding="utf-8")

t1=df_all['id'].map(lambda x: int(x<163800))
t2=df_all['id'].map(lambda x: int(x>206650))
t3=df_all['id'].map(lambda x: int(x<163800) or int(x>221473))

df_train = pd.read_csv(DATA_DIR+'/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv(DATA_DIR+'/test.csv', encoding="ISO-8859-1")

num_train = df_train.shape[0]
y = df_all["relevance"][:num_train]
id_test=df_all["id"][num_train:]

t1_tr=t1.iloc[:num_train]
t2_tr=t2.iloc[:num_train]
t3_tr=t3.iloc[:num_train]
t1_tt=pd.DataFrame(t1.iloc[num_train:])
t2_tt=pd.DataFrame(t2.iloc[num_train:])
t3_tt=pd.DataFrame(t3.iloc[num_train:])
t1_tt.index=range(len(t1_tt))
t2_tt.index=range(len(t2_tt))
t3_tt.index=range(len(t3_tt))



train=pd.concat([train,  t3_tr ], axis=1)
test=pd.concat([test,  t3_tt ], axis=1)

#rename columns
train.columns=range(len(train.keys()))
test.columns=range(len(test.keys()))
#train["relevance"]=y["relevance"]
train["relevance"]=y

trainX=train
y_tr = trainX['relevance'].values
X_tr = trainX.drop(['relevance'],axis=1).values




from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics
from scipy.optimize import nnls


class MLR(object):
    def __init__(self):
        self.coef_ = 0
  
    def fit(self, X, y):
        self.coef_ = sp.optimize.nnls(X, y)[0]
        self.coef_ = np.array(map(lambda x: x/sum(self.coef_), self.coef_))
  
    def predict(self, X):
        predictions = np.array(map(sum, self.coef_ * X))
        return predictions

#selecting stacker model
n_folds=5
skf = list(StratifiedKFold(y_tr, n_folds, shuffle=True))
blend_train = np.zeros((X_tr.shape[0]))
#clf=MLR()  
clf = LinearRegression()
#clf = neighbors.KNeighborsRegressor(128, weights="uniform", leaf_size=5)

#select first model
mn_rmse=1
model_n=0
for i in range(0,len(train.keys())-1):
     for j, (train_index, cv_index) in enumerate(skf):
            #print 'Fold [%s]' % (j)
            
            # This is the training and validation set
            X_train = X_tr[:,i][train_index]
            Y_train = y_tr[train_index]
            X_cv = X_tr[:,i][cv_index]
            Y_cv = y_tr[cv_index]
            X_train=X_train.reshape((len(X_train),1))
            Y_train=Y_train.reshape((len(Y_train),1))
            X_cv=X_cv.reshape((len(X_cv),1))
            Y_cv=Y_cv.reshape((len(Y_cv),1))
            clf.fit(X_train,Y_train)
            blend_train[cv_index] = clf.predict(X_cv)
            
     if sqrt(metrics.mean_squared_error(y_tr, blend_train))<mn_rmse:
            mn_rmse=sqrt(metrics.mean_squared_error(y_tr, blend_train))
            print i, mn_rmse
            model_n=i
     #print i, sqrt(metrics.mean_squared_error(y_tr, blend_train))
model_list=list()
model_list.append(model_n)


model_collection=X_tr[:,model_n]
model_collection=np.vstack((model_collection)).T
cur_mn=mn_rmse


#select other models
for j in range(len(train.keys())-1):
    pred_mn_rmse=cur_mn
    for i in range(len(train.keys())-1):
        
        if (i in model_list):
            OK="OK"
        else:
            for k, (train_index, cv_index) in enumerate(skf):

                
                # This is the training and validation set
                X_train = X_tr[:,i][train_index]
                Y_train = y_tr[train_index]
                X_cv = X_tr[:,i][cv_index]
                Y_cv = y_tr[cv_index]


                CV_m=model_collection[0][train_index]
                for it in range(1,len(model_collection)):
                    tmp=model_collection[it][train_index]
                    CV_m=np.vstack((CV_m,tmp))

                clf.fit(np.vstack((CV_m,X_train)).T, Y_train)
                #clf.fit(X_train,Y_train)
                CV_n=model_collection[0][cv_index]
                for it in range(1,len(model_collection)):
                    tmp=model_collection[it][cv_index]
                    CV_n=np.vstack((CV_n,tmp))
                blend_train[cv_index] = clf.predict(np.vstack((CV_n,X_cv)).T)

            if sqrt(metrics.mean_squared_error(y_tr, blend_train))<cur_mn:
                cur_mn = sqrt(metrics.mean_squared_error(y_tr, blend_train))
                model_n=i
    if (model_list[len(model_list)-1]==model_n) or abs(cur_mn-pred_mn_rmse)<0.00001:
        break
    model_list.append(model_n)
    model_collection=np.vstack((model_collection,X_tr[:,model_n]))

    print model_list
    print cur_mn


print len(model_list)    

#choose top12 models     
model_list2=model_list[0:12]
test_fin=test[model_list2]
train_fin=train[model_list2]

#select model for stacking
clf = Ridge(alpha=3.0)
clf.fit(train_fin, y)

pred1 = clf.predict(test_fin)
pred1[pred1<1.]=1.
pred1[pred1>3.]=3.


 
#saved_results
pd.DataFrame({"id": id_test, "relevance": pred1}).to_csv(MODELS_DIR+"/submissions_ensemble_n_models_from_m_11_04_2016.csv",index=False)
   

#X_new=train_fin
#import statsmodels.api as sm
#X_new = sm.add_constant(  X_new  )
#results = sm.OLS(y, X_new).fit()
#print results.summary()