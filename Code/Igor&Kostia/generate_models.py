# -*- coding: utf-8 -*-
"""
Generating models: Igor's part.

Competition: HomeDepot Search Relevance
Author: Igor Buinyi
Team: Turing test
"""

from config_IgorKostia import *

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from time import time
import re
import os
from scipy.stats import pearsonr


df_train = pd.read_csv(DATA_DIR+'/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv(DATA_DIR+'/test.csv', encoding="ISO-8859-1")
num_train = df_train.shape[0] #number of observations
num_test = df_test.shape[0] #number of observations


#df_all = pd.read_csv(FEATURES_DIR+'/df_basic_features_wo_google.csv', encoding="utf-8")
df_all = pd.read_csv(FEATURES_DIR+'/df_basic_features.csv', encoding="utf-8")

df_dist = pd.read_csv(FEATURES_DIR+'/df_dist_new.csv', encoding="utf-8")
df_st_tfidf= pd.read_csv(FEATURES_DIR+'/df_st_tfidf.csv', encoding="utf-8")
if 'Unnamed: 0' in df_st_tfidf.keys():
    df_st_tfidf = df_st_tfidf.drop(['Unnamed: 0'],axis=1)
df_tfidf_intersect = pd.read_csv(FEATURES_DIR+'/df_tfidf_intersept_new.csv', encoding="utf-8")
df_word2vec = pd.read_csv(FEATURES_DIR+'/df_word2vec_new.csv', encoding="utf-8")
df_dld = pd.read_csv(FEATURES_DIR+'/dld_features.csv', encoding="utf-8")

#df_above15 = pd.read_csv(FEATURES_DIR+'/df_feature_above15_wo_google.csv', encoding="utf-8")
df_above15 = pd.read_csv(FEATURES_DIR+'/df_feature_above15_ext.csv', encoding="utf-8")
df_all = pd.merge(df_all, df_above15, how='left', on='id')

df_all = pd.merge(df_all, df_dist, how='left', on='id')
df_all = pd.merge(df_all, df_st_tfidf, how='left', on='id')
df_all = pd.merge(df_all, df_tfidf_intersect, how='left', on='id')
df_all = pd.merge(df_all, df_word2vec, how='left', on='id')
df_all = pd.merge(df_all, df_dld, how='left', on='id')


#df_bm_dummy = pd.read_csv(FEATURES_DIR+'/df_brand_material_dummies_wo_google.csv', encoding="utf-8")
#df_thekey_dummy = pd.read_csv(FEATURES_DIR+'/df_thekey_dummies_wo_google.csv', encoding="utf-8")
df_bm_dummy = pd.read_csv(FEATURES_DIR+'/df_brand_material_dummies.csv', encoding="utf-8")
df_thekey_dummy = pd.read_csv(FEATURES_DIR+'/df_thekey_dummies.csv', encoding="utf-8")
df_all = pd.merge(df_all, df_bm_dummy, how='left', on='id')
df_all = pd.merge(df_all, df_thekey_dummy, how='left', on='id')


#create dummy
df_all['id_dummy']=df_all['id'].map(lambda x: int(x>163700 and x<=221473))

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

### load feature importances from benchmark model to drop some features
df_importance = pd.read_csv(MODELS_DIR+'/feature_importances_benchmark_without_dummies.csv', encoding="utf-8")
df_importance=df_importance.sort_values(['importance'],ascending=[0])
df_importance['cumulative']=df_importance['importance'].map(lambda x: sum(df_importance['importance'][df_importance['importance']>=x]))
important_var_list=list(df_importance['name'][df_importance['cumulative']<0.9990])
#var_list.remove('product_uid')

imp_THRESHOLD=0.990
variance_THRESHOLD=0.95
new_var_list=[list(df_importance['name'][df_importance['cumulative']<imp_THRESHOLD])[0]]
for cnt in range(1,len(list(df_importance['name'][df_importance['cumulative']<imp_THRESHOLD]))):
    var=list(df_importance['name'][df_importance['cumulative']<imp_THRESHOLD])[cnt]
    max_abs_corr=0
    for var1 in new_var_list:
        corr=abs(pearsonr(df_all[var],df_all[var1])[0])
        if corr>max_abs_corr:
            max_abs_corr=corr
    if max_abs_corr< variance_THRESHOLD:
        new_var_list.append(var)
    if cnt % 10 ==0:
        print cnt, len(new_var_list)

### load feature importances from benchmark model to drop some dummies
df_importance_dummy = pd.read_csv(MODELS_DIR+'/feature_importances_benchmark_top40_and_dummies.csv', encoding="utf-8")
df_importance_dummy=df_importance_dummy.sort_values(['importance'],ascending=[0])
df_importance_dummy['cumulative']=df_importance_dummy['importance'].map(lambda x: sum(df_importance_dummy['importance'][df_importance_dummy['importance']>=x]))
important_dummy_list=list(df_importance_dummy['name'][df_importance_dummy.apply(lambda x: x['cumulative']<0.99 \
    and x['name'][0]!=x['name'][0].lower(),axis=1)])
new_dummy_list=list(df_importance_dummy['name'][df_importance_dummy.apply(lambda x: x['cumulative']<0.96 \
    and x['name'][0]!=x['name'][0].lower(),axis=1)])
    

print len(important_var_list), len(new_var_list)
print len(important_dummy_list), len(new_dummy_list)


important_vars=['id','relevance','id_dummy','above15_dummy_frequency_of_beforethekey_thekey','description20_percentile']+important_var_list+important_dummy_list
selected_vars=['id','relevance','id_dummy','above15_dummy_frequency_of_beforethekey_thekey','description20_percentile']+new_var_list+new_dummy_list

print len(important_vars)
print len(selected_vars)

all_vars= list(df_all.keys())
print len(all_vars)






#######################################


from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

GB_params1 = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 1, 'min_samples_leaf':15,
               'learning_rate': 0.035, 'loss': 'ls', 'verbose':0, 'random_state':2016}
GB_params2 = {'n_estimators': 900, 'max_depth': 6, 'min_samples_split': 1, 'min_samples_leaf':12,
               'learning_rate': 0.02, 'loss': 'ls', 'verbose':0, 'random_state':2017}
xgb_params1={'colsample_bytree': 1, 'silent': 1, 'nthread': 8, 'min_child_weight': 12,\
 'n_estimators': 165, 'subsample': 1, 'learning_rate': 0.08, 'objective': 'reg:linear',\
 'seed': 2016, 'max_depth': 6, 'gamma': 0.}
 
xgb_params2={'colsample_bytree': 1, 'silent': 1, 'nthread': 8, 'min_child_weight': 10,\
 'n_estimators': 300, 'subsample': 1, 'learning_rate': 0.09, 'objective': 'reg:linear',\
 'seed': 10, 'max_depth': 7, 'gamma': 0.}
 

xgb_params3={'colsample_bytree': 0.5, 'silent': 1, 'nthread': 8, 'min_child_weight': 12, \
   'n_estimators': 500, 'subsample': 0.7, 'learning_rate': 0.025, 'objective': 'reg:linear',  \
   'seed': 11, 'max_depth': 6, 'gamma': 0.2} 
 
rfr_params1={'n_estimators':100, 'max_depth':15, 'min_samples_leaf':12, 'max_features':0.55,
            'min_samples_split':1, 'n_jobs': -1, 'random_state':2016 }
            
            
xtree_params={'n_estimators': 250,  'max_depth': None, 'min_samples_split':12, \
              'verbose': 1, 'random_state':2016, 'n_jobs':-1}

#dt_params1 = {'max_depth':6, 'min_samples_split':12, 
#             'min_samples_leaf':5, 'min_weight_fraction_leaf':0.0, 'max_features':200, 
#             'random_state':2016, 'max_leaf_nodes':None} 
#clf = DecisionTreeRegressor(**dt_params1)
#RMSE  above 0.46



clf_list=[(RandomForestRegressor(**rfr_params1), selected_vars, "RF_params1_selected"),
          (ExtraTreesRegressor(**xtree_params), selected_vars, "extratrees_selected"),
          (GradientBoostingRegressor(**GB_params1), important_vars, "GB_params1_important"),
          (SVR(C=.2, kernel='rbf'), selected_vars, "SVR_C0_2_kernel_rbf_selected"),
          (BaggingRegressor(xgb.XGBRegressor(**xgb_params2), n_estimators=10, 
                       random_state=np.random.RandomState(2016)), important_vars, "xgboost2+bagging10_important"),
          (BaggingRegressor(xgb.XGBRegressor(**xgb_params3), n_estimators=10, 
                       random_state=np.random.RandomState(2016)), selected_vars, "xgboost3+bagging10_selected"),
          (GradientBoostingRegressor(**GB_params2), selected_vars, "GB_params2_selected")
 ]



id_train = df_all['id'].iloc[:num_train]
id_test = df_all['id'].iloc[num_train:]
y_train = df_all['relevance'].iloc[:num_train].values

t00=time()
## Used StratifiedKFOLD(n_folds=3), 1 fold is used for training, two folds for validation.
## Two splits are used to use check the results and ensure robustness.
skf = list(StratifiedKFold(y_train, n_folds=3, shuffle=True,random_state=2016)) \
    + list(StratifiedKFold(y_train, n_folds=3, shuffle=True,random_state=2017))


for clf, feature_list, name_str in clf_list:
    print clf
    print 'Model', name_str
    if "SVR" in name_str:
        X_matrix = preprocessing.scale(df_all[feature_list].drop(['id','relevance'],axis=1),axis=0)
    else:
        X_matrix = df_all[feature_list].drop(['id','relevance'],axis=1).values

    X_train = X_matrix[:num_train]
    X_test = X_matrix[num_train:]

    print '\tStep 0: Cross validation'
    cv_label_pred_stacked=[]
    cv_indices_stacked=[]
    cv_labels_stacked=[]
    
    t0 = time()
    
        
    total_RMSE=0
    """    
    generate prediction for cross validation folds
    """
    for i, (cv_test_indices, cv_train_indices) in enumerate(skf):
        assert len(cv_test_indices)>1.5*len(cv_train_indices)
        
        cv_features_train =X_train[cv_train_indices]
        cv_features_test =X_train[cv_test_indices]
        cv_labels_train =y_train[cv_train_indices]
        cv_labels_test =y_train[cv_test_indices]
        
        clf.fit(cv_features_train, cv_labels_train)
        
        cv_label_pred=clf.predict(cv_features_test)
        
        
        cv_label_pred_stacked = np.concatenate((cv_label_pred_stacked, cv_label_pred))
        cv_indices_stacked = np.concatenate((cv_indices_stacked, cv_test_indices)) 
        cv_labels_stacked = np.concatenate((cv_labels_stacked, cv_labels_test))  
        
        
        RMSE = mean_squared_error(cv_labels_test, cv_label_pred)**0.5
        total_RMSE += RMSE
        print '\t\tFold [%s] [RMSE: %s] [%s minutes]' % (i,round(RMSE,6), round((time()-t0)/60,1))
    
    pd.DataFrame({"id": id_train[cv_indices_stacked], "predicted": cv_label_pred_stacked, \
        "actual": cv_labels_stacked}).to_csv(MODELSENSEMBLE_DIR+'/trainvalidation_'+name_str+'_2015-04-23.csv',index=False)
    print '\tTrain validation file saved [RMSE: %s]' % (round(total_RMSE/len(skf),6))
    
    """
    generate predictions for test
    """
    print '\tStep 1: Predict test labels'
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(MODELSENSEMBLE_DIR+'/testprediction_'+name_str+'_2015-04-23.csv',index=False)
    print '\tTest prediction file saved [%s minutes]\n'  % (round((time()-t0)/60,1))
    
print 'Total time %s minutes' % (round((time()-t00)/60,1))
                
