# -*- coding: utf-8 -*-
"""
Generating ensemble output from models: Igor's part.

Competition: HomeDepot Search Relevance
Author: Igor Buinyi
Team: Turing test
"""

from config_IgorKostia import *

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from time import time
import re
import os
from scipy.stats import pearsonr


df_train = pd.read_csv(DATA_DIR+'/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv(DATA_DIR+'/test.csv', encoding="ISO-8859-1")
num_train = df_train.shape[0] #number of observations
num_test = df_test.shape[0] #number of observations




dir_name=MODELSENSEMBLE_DIR


files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]


### Load files
models=[]
df_all_validation=None
#filename_from_which_id_is_read=None
for f in files:
    if f.endswith(".csv") and f.startswith("trainvalidation_"):
        model_name=f.replace("trainvalidation_","").replace(".csv","")
        submission_file=f.replace("trainvalidation_","testprediction_")
        if os.path.exists(os.path.join(dir_name, "testprediction_"+model_name+".csv")):
            df_model_validation = pd.read_csv(os.path.join(dir_name, f), index_col=False)
            df_model_submission = pd.read_csv(os.path.join(dir_name, submission_file), index_col=False)
            
            if df_all_validation is None:
                df_all_validation=df_model_validation[['id','actual']]
                df_all_submission=df_model_submission[['id']]
                filenames_from_which_ids_are_read=\
                    {'validation':f,
                     'submission':f.replace("trainvalidation_","").replace(".csv","")}
                
            if sum(df_all_validation['id']!=df_model_validation['id'])>0:
                raise ValueError("'id' column in file\n\t"+f+
                "\nis different from file \n\t"+filenames_from_which_ids_are_read['validation'])
            elif sum(df_all_validation['actual']!=df_model_validation['actual'])>0:
                raise ValueError("'actual' column in file\n\t"+f+
                "\nis different from file \n\t"+filenames_from_which_ids_are_read['validation'])
            else:
                df_all_validation[model_name]=df_model_validation['predicted']  
                
            if sum(df_all_submission['id']!=df_model_submission['id'])>0:
                raise ValueError("'id' column in file\n\t"+submission_file+
                "\nis different from file \n\t"+filenames_from_which_ids_are_read['submission'])
            else:
                df_all_submission[model_name]=df_model_submission['relevance']  
                
            models.append(model_name)
            print "loaded", model_name

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm



n=len(df_all_validation['actual'])/2

for model in models:
    print "%s\t1st split: %.5f, 2nd split: %.5f, total: %.5f" % (model, \
    mean_squared_error(df_all_validation['actual'][:n], df_all_validation[model][:n])**0.5, \
    mean_squared_error(df_all_validation['actual'][n:], df_all_validation[model][n:])**0.5,\
    mean_squared_error(df_all_validation['actual'], df_all_validation[model])**0.5)

mean_relevance=np.mean(df_all_validation['actual'])
y = df_all_validation['actual'].values
X = sm.add_constant(df_all_validation[models].apply(lambda x: x-mean_relevance))


def adjusted_predictions(pred):
    pred[pred<1.]=1.
    pred[pred>3.]=3.
    return pred


"""
# Remove redundant models
models.remove('xgboost1+bagging10_all_2015-04-21')
models.remove('extratrees_selected_wo_google_2015-04-21')
models.remove('RF_params1_important_wo_google_2015-04-21')
models.remove('xgboost3_selected_2015-04-23')
"""

""" 
# Correlations
aa=df_all_validation[models].corr()
bb=df_all_submission[models].corr()
plt.matshow(df_all_validation[models].corr())
plt.matshow(df_all_submission[models].corr())
aa.to_csv(os.path.join(dir_name,'correlations_val.csv'))
bb.to_csv(os.path.join(dir_name,'correlations_subm.csv'))


## separate CV result into two parts and check the resutls
n=len(df_all_validation['actual'])/2


results_fold1 = sm.OLS(y[:n], X[:n]).fit()
#print results_fold1.summary()
X_fold2 = X[n:]
pred_fold2=adjusted_predictions(results_fold1.predict(X_fold2))
print mean_squared_error(df_all_validation['actual'][:n], adjusted_predictions(results_fold1.predict(X[:n])))**0.5
print mean_squared_error(df_all_validation['actual'][n:], pred_fold2)**0.5

results_fold2 = sm.OLS(y[n:], X[n:]).fit()
#print results_fold1.summary()
X_fold1 = X[:n]
pred_fold1=adjusted_predictions(results_fold2.predict(X_fold1))
print mean_squared_error(df_all_validation['actual'][n:], adjusted_predictions(results_fold2.predict(X[n:])))**0.5
print mean_squared_error(df_all_validation['actual'][:n], pred_fold1)**0.5


results_fold1 = sm.OLS(y[:n], X_with_squares[:n]).fit()
#print results_fold1.summary()
X_fold2 = X_with_squares[n:]
pred_fold2=adjusted_predictions(results_fold1.predict(X_fold2))
print mean_squared_error(df_all_validation['actual'][:n], adjusted_predictions(results_fold1.predict(X_with_squares[:n])))**0.5
print mean_squared_error(df_all_validation['actual'][n:], pred_fold2)**0.5

"""


### regress train relevances on CV prediction from models
results = sm.OLS(y, X).fit()
print results.summary()
print mean_squared_error(df_all_validation['actual'], results.predict(X))**0.5

X_test = sm.add_constant(df_all_submission[models].apply(lambda x: x-mean_relevance))
pred1=results.predict(X_test)
pred1[pred1<1.]=1.
pred1[pred1>3.]=3.

#generate final ensemble from Igor
id_test= df_all_submission['id']
pd.DataFrame({"id": id_test, "relevance": pred1}).to_csv(MODELS_DIR+'/submission_2016-04-23_ensemble_8models_Igor_final.csv',index=False)




#                            OLS Regression Results                            
#==============================================================================
#Dep. Variable:                      y   R-squared:                       0.335
#Model:                            OLS   Adj. R-squared:                  0.335
#Method:                 Least Squares   F-statistic:                 1.862e+04
#Date:                Sat, 23 Apr 2016   Prob (F-statistic):               0.00
#Time:                        14:03:32   Log-Likelihood:            -1.7416e+05
#No. Observations:              296268   AIC:                         3.483e+05
#Df Residuals:                  296259   BIC:                         3.484e+05
#Df Model:                           8                                         
#Covariance Type:            nonrobust                                         
#=====================================================================================================================
#                                                        coef    std err          t      P>|t|      [95.0% Conf. Int.]
#---------------------------------------------------------------------------------------------------------------------
#const                                                 2.3751      0.001   2759.954      0.000         2.373     2.377
#extratrees_selected_2015-04-21                        0.2040      0.008     24.171      0.000         0.187     0.221
#GB_params1_important_2015-04-21                       0.0550      0.019      2.901      0.004         0.018     0.092
#GB_params2_selected_2015-04-21                        0.1541      0.019      8.063      0.000         0.117     0.192
#RF_params1_selected_2015-04-21                       -0.1099      0.014     -7.940      0.000        -0.137    -0.083
#SVR_C0_2_kernel_rbf_selected_2015-04-21               0.1737      0.007     24.146      0.000         0.160     0.188
#xgboost2+bagging10_important_2015-04-21               0.1700      0.019      9.143      0.000         0.134     0.206
#xgboost3+bagging10_important_wo_google_2015-04-21     0.2415      0.014     16.916      0.000         0.213     0.269
#xgboost3+bagging10_selected_2015-04-21                0.1489      0.026      5.796      0.000         0.099     0.199
#==============================================================================
#Omnibus:                     8069.021   Durbin-Watson:                   1.983
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):             8742.310
#Skew:                          -0.418   Prob(JB):                         0.00
#Kurtosis:                       3.090   Cond. No.                         36.6
#==============================================================================
#
#Warnings:
#[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



