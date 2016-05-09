# -*- coding: utf-8 -*-
"""
The file to generated feature importances from the benchmark Gradient Boost model: 
separately for dummies and all other features.

Competition: HomeDepot Search Relevance
Author: Igor Buinyi
Team: Turing test
"""

from config_IgorKostia import *

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from time import time


# get num_tain
df_train = pd.read_csv(DATA_DIR+'/train.csv', encoding="ISO-8859-1")
num_train = df_train.shape[0] #number of observations

# load features
df_all = pd.read_csv(FEATURES_DIR+'/df_basic_features.csv', encoding="utf-8")
df_dist = pd.read_csv(FEATURES_DIR+'/df_dist_new.csv', encoding="utf-8")
df_st_tfidf= pd.read_csv(FEATURES_DIR+'/df_st_tfidf.csv', encoding="utf-8")
if 'Unnamed: 0' in df_st_tfidf.keys():
    df_st_tfidf = df_st_tfidf.drop(['Unnamed: 0'],axis=1)
df_tfidf_intersect = pd.read_csv(FEATURES_DIR+'/df_tfidf_intersept_new.csv', encoding="utf-8")
df_word2vec = pd.read_csv(FEATURES_DIR+'/df_word2vec_new.csv', encoding="utf-8")
df_dld = pd.read_csv(FEATURES_DIR+'/dld_features.csv', encoding="utf-8")

"""
the following features and files were added later
so this is the adjustment in order to reproduce the same results
"""
df_above15 = pd.read_csv(FEATURES_DIR+'/df_feature_above15_ext.csv', encoding="utf-8")
df_above15 = df_above15[['id','above15_dummy_frequency_of_beforethekey_thekey']]
df_all = pd.merge(df_all, df_above15, how='left', on='id')

# merge
df_all = pd.merge(df_all, df_dist, how='left', on='id')
df_all = pd.merge(df_all, df_st_tfidf, how='left', on='id')
df_all = pd.merge(df_all, df_tfidf_intersect, how='left', on='id')
df_all = pd.merge(df_all, df_word2vec, how='left', on='id')
df_all = pd.merge(df_all, df_dld, how='left', on='id')


# drop product_uid and some vars
drop_list=['product_uid']
drop_list+=['description_similarity_10',	'description_similarity_11-20',	'description_similarity_30',
            'description_similarity_21-30', 'description_similarity_10rel', 'description_similarity_11-20rel',
            'description_similarity_30rel',	'description_similarity_21-30rel', 'description_similarity_21-30to10',
            'word_in_title_string_only_num',	'word_in_title_string_only_sum',	'word_in_title_string_only_let']


print len(df_all.keys())
new_drop_list=[]
for var in drop_list:
    if var in df_all.keys():
        new_drop_list.append(var)
df_all=df_all.drop(new_drop_list,axis=1)
print len(df_all.keys())


# generate matrices to be used in clf
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
id_train = df_train['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values


#########################################################################
##### use GradientBoostingRegressor to generate feature importances
t0 = time()
params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 1, 'min_samples_leaf':15, 'learning_rate': 0.035, 'loss': 'ls', 'verbose':1}
clf = GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred[y_pred<1.]=1.
y_pred[y_pred>3.]=3.


pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(MODELS_DIR+'/submission_benchmark_without_dummies.csv',index=False)
sorted_idx = np.argsort(clf.feature_importances_)
pd.DataFrame({"name":df_all.keys().drop(['id','relevance'])[sorted_idx], "importance": clf.feature_importances_[sorted_idx]}).to_csv(MODELS_DIR+'/feature_importances_benchmark_without_dummies.csv',index=False)

print "file saved"
print 'modelling time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


#### load feature importances from file
df_importance = pd.read_csv(MODELS_DIR+'/feature_importances_benchmark_without_dummies.csv', encoding="utf-8")
df_importance=df_importance.sort_values(['importance'],ascending=[0])
df_importance['cumulative']=df_importance['importance'].map(lambda x: sum(df_importance['importance'][df_importance['importance']>=x]))
var_list=list(df_importance['name'][df_importance['cumulative']<0.990])


# use only 40 vars in the next step
df_all=df_all[['id','relevance']+var_list[0:40]]


# load dummies
df_bm_dummy = pd.read_csv(FEATURES_DIR+'/df_brand_material_dummies.csv', encoding="utf-8")
df_thekey_dummy = pd.read_csv(FEATURES_DIR+'/df_thekey_dummies.csv', encoding="utf-8")
df_all = pd.merge(df_all, df_bm_dummy, how='left', on='id')
df_all = pd.merge(df_all, df_thekey_dummy, how='left', on='id')

# generate matrices to be used in clf
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
id_train = df_train['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

#################################################################################
##### use GradientBoostingRegressor to generate feature importances for dummies
t0 = time()
params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 1, 'min_samples_leaf':15, 'learning_rate': 0.035, 'loss': 'ls', 'verbose':1}
clf = GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred[y_pred<1.]=1.
y_pred[y_pred>3.]=3.


pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(MODELS_DIR+'/submission_benchmark_top40_and_dummies.csv',index=False)
sorted_idx = np.argsort(clf.feature_importances_)
pd.DataFrame({"name":df_all.keys().drop(['id','relevance'])[sorted_idx], "importance": clf.feature_importances_[sorted_idx]}).to_csv(MODELS_DIR+'/feature_importances_benchmark_top40_and_dummies.csv',index=False)

print "file saved"
print 'modelling time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

