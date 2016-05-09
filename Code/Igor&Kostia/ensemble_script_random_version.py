# -*- coding: utf-8 -*-
"""

Code for training and saving models on random part of the features.
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
from sklearn.neural_network import BernoulliRBM
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from math import sqrt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn import preprocessing


drop_list= []





#loading data
df_all = pd.read_csv(FEATURES_DIR+'/df_basic_features.csv', encoding="utf-8")
df_all1 = pd.read_csv(FEATURES_DIR+'/df_thekey_dummies.csv', encoding="utf-8")
df_all2 = pd.read_csv(FEATURES_DIR+'/df_brand_material_dummies.csv', encoding="utf-8")

df_dld = pd.read_csv(FEATURES_DIR+'/dld_features.csv', encoding="utf-8")
df_tfidf_st = pd.read_csv(FEATURES_DIR+'/df_st_tfidf.csv', encoding="utf-8")
df_word2vec = pd.read_csv(FEATURES_DIR+'/df_word2vec_new.csv', encoding="utf-8")
df_dist_new = pd.read_csv(FEATURES_DIR+'/df_dist_new.csv', encoding="utf-8")
df_tfidf_intersept_new = pd.read_csv(FEATURES_DIR+'/df_tfidf_intersept_new.csv', encoding="utf-8")



df_all = pd.merge(df_all, df_dld, how='left', on='id')
df_all = pd.merge(df_all, df_tfidf_st, how='left', on='id')
df_all = pd.merge(df_all, df_word2vec, how='left', on='id')
df_all = pd.merge(df_all, df_dist_new, how='left', on='id')
df_all = pd.merge(df_all, df_tfidf_intersept_new, how='left', on='id')
#df_all = pd.merge(df_all, df_all1, how='left', on='id')
#df_all = pd.merge(df_all, df_all2, how='left', on='id')

all_names=df_all.keys()
print len(df_all.keys())
list_of_features=df_all.keys()
new_list=[] 
for var in list_of_features:
     if 'uniq_'  in var:
         new_list.append(var)
new_list.append('product_uid')
new_list.append("Unnamed: 0")        
df_all=df_all.drop(list(set(list(all_names & new_list))),axis=1)


list_of_features=df_all.keys().drop(["relevance","id"])

df_train = pd.read_csv(DATA_DIR+'/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv(DATA_DIR+'/test.csv', encoding="ISO-8859-1")

num_train = df_train.shape[0]

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']


#df_train = df_train.iloc[np.random.permutation(len(df_train))]

y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

X1=X_train
Y1=y_train
X2=X_test

X = np.vstack((X1,X2))
X = preprocessing.scale(X,axis=0)
X1 = X[:num_train]
X2 = X[num_train:]



def run(X,Y,X2,bclf,iteration):
      
    
    t00=time()
    dev_cutoff = len(Y) #* 9/10
    X_dev = X[:dev_cutoff]
    Y_dev = Y[:dev_cutoff]

    X_test = X2

    
    n_trees = 10
    n_folds = 3
    n_features=X_dev.T.shape[0]
    xgb_params0={'colsample_bytree': 1, 'silent': 1, 'nthread': 8, 'min_child_weight': 10,\
    'n_estimators': 300, 'subsample': 1, 'learning_rate': 0.09, 'objective': 'reg:linear',\
    'seed': 10, 'max_depth': 7, 'gamma': 0.}
    xgb_params1={'colsample_bytree': 0.77, 'silent': 1, 'nthread': 8, 'min_child_weight': 15,\
    'n_estimators': 500, 'subsample': 0.77, 'learning_rate': 0.035, 'objective': 'reg:linear',\
    'seed': 11, 'max_depth': 6, 'gamma': 0.2}
 #     Our level 0 classifiers
    clfs = [
        ExtraTreesRegressor(n_estimators = n_trees * 20),
        BaggingRegressor(base_estimator=xgb.XGBRegressor(**xgb_params0), n_estimators=10, random_state=np.random.RandomState(2016) ),
        RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=6, max_features=0.9,\
           min_samples_split=1, n_jobs= -1, random_state=2014),
        AdaBoostRegressor(base_estimator=None, n_estimators=250, learning_rate=0.03, loss='linear', random_state=20160703),
        BaggingRegressor(base_estimator=None, n_estimators=200, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0),
        neighbors.KNeighborsRegressor(128, weights="uniform", leaf_size=5),
        SVR(kernel='rbf', C=0.2, gamma=0.1),
        SVR(kernel='rbf', C=0.3, gamma=0.5),
        SVR(kernel='linear', C=0.2),
        SVR(kernel='poly', C=0.2, degree=2),
        GradientBoostingRegressor(n_estimators=500, max_depth=6, min_samples_split=1, min_samples_leaf=15, learning_rate=0.035, loss='ls',random_state=10),
        xgb.XGBRegressor(**xgb_params0),
        xgb.XGBRegressor(**xgb_params1),
        DecisionTreeRegressor(criterion='mse', splitter='random', max_depth=4, min_samples_split=7, min_samples_leaf=30, min_weight_fraction_leaf=0.0, max_features='sqrt', random_state=None, max_leaf_nodes=None, presort=False)
    ]
   
    # Ready for cross validation
    skf = list(StratifiedKFold(Y_dev, n_folds, shuffle=True))
    blend_train = np.zeros((X_dev.shape[0], len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((X_test.shape[0], len(clfs))) # Number of testing data x Number of classifiers
     
    print 'X_test.shape = %s' % (str(X_test.shape))
    print 'blend_train.shape = %s' % (str(blend_train.shape))
    print 'blend_test.shape = %s' % (str(blend_test.shape))
    
    # For each classifier, we train the number of fold times (=len(skf))
    for j, clf in enumerate(clfs):
        print 'Training classifier [%s]' % (clf) 
        print 'Training classifier [%s]' % ((j+1.0)/len(clfs))
        t0=time()
        for i, (train_index, cv_index) in enumerate(skf):
            print 'Fold [%s]' % (i)
            
            # This is the training and validation set
            X_train = X_dev[train_index]
            Y_train = Y_dev[train_index]
            X_cv = X_dev[cv_index]
            Y_cv = Y_dev[cv_index]
            
            clf.fit(X_train, Y_train)
            
            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            blend_train[cv_index, j] = clf.predict(X_cv)
            print sqrt(metrics.mean_squared_error(Y_cv, blend_train[cv_index, j]))
        print sqrt(metrics.mean_squared_error(Y_dev, blend_train[:,j]))
        
        clf.fit(X_dev, Y_dev)
        blend_test[:, j]=clf.predict(X_test)
        print 'train time:',round(time()-t0,3) ,'s\n'
    print 'Y_dev.shape = %s' % (Y_dev.shape)
    
    bclf.fit(blend_train, Y_dev)
    Y_test_predict = bclf.predict(blend_test)
    print 'all time:',round(time()-t00,3) ,'s\n'
    
    return Y_test_predict, blend_test, blend_train, Y_dev



    # run many times to get a better result, it's not quite stable.
for i in range(100,130):
        print 'Iteration [%s]' % (i)
        
        bclf = LinearRegression()
        tmp_list=list_of_features
        R=int(round(npr.random_sample(size=None)*len(list_of_features)))
        rand_order=npr.permutation(len(tmp_list))
        tmp_list = tmp_list[rand_order]
        first_part=tmp_list[:R]
        second_part=tmp_list[R:]
        FP=rand_order[:R]
        SP=rand_order[R:]
        X1_f=X1[:,FP]
        X1_s=X1[:,SP]
        X2_f=X2[:,FP]
        X2_s=X2[:,SP]
        if R<20:
            R=21
        if R>len(list_of_features)-20:
            R= len(list_of_features) - 21
        print R
        pred, bx_test, bx_train, by_train = run(X1_f,Y1, X2_f,bclf,"first"+str(i)+"")
        pd.DataFrame({"number": FP, "feature_name": first_part}).to_csv(MODELS_DIR+"/first_part_"+str(i)+".csv",index=False) 
        pd.DataFrame(bx_train).to_csv(MODELS_DIR+"/train_first_"+str(i)+".csv",index=False) 
        pd.DataFrame(bx_test).to_csv(MODELS_DIR+"/test_first_"+str(i)+".csv",index=False)
        
        pred, bx_test, bx_train, by_train = run(X1_s,Y1, X2_s,bclf,"second"+str(i)+"")
        pd.DataFrame({"number": FP, "feature_name": first_part}).to_csv(MODELS_DIR+"/second_part_"+str(i)+".csv",index=False) 
        pd.DataFrame(bx_train).to_csv(MODELS_DIR+"/train_second_"+str(i)+".csv",index=False) 
        pd.DataFrame(bx_test).to_csv(MODELS_DIR+"/test_second_"+str(i)+".csv",index=False)

