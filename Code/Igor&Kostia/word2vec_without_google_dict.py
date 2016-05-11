# -*- coding: utf-8 -*-
"""

The same code as word2vec.py, but different input data and only two models instead four.
Competition: HomeDepot Search Relevance
Author: Kostia Omelianchuk
Team: Turing test

"""


from config_IgorKostia import *

import gensim
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
import nltk
from time import time
import re
import os
import math as m
import pandas as pd
from gensim import models


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
df_all=pd.read_csv(PROCESSINGTEXT_DIR+"/df_train_and_test_processed_wo_google.csv", encoding="ISO-8859-1")
df_all1=pd.read_csv(PROCESSINGTEXT_DIR+"/df_product_descriptions_processed_wo_google.csv", encoding="ISO-8859-1")
df_all2 = pd.merge(df_all, df_all1, how="left", on="product_uid")
df_all = df_all2
df_all1=pd.read_csv(PROCESSINGTEXT_DIR+"/df_attribute_bullets_processed_wo_google.csv", encoding="ISO-8859-1")
df_all2 = pd.merge(df_all, df_all1, how="left", on="product_uid")
df_all = df_all2
df_attr = pd.read_csv(PROCESSINGTEXT_DIR+'/df_attributes_kostia.csv', encoding="ISO-8859-1")
df_all = pd.merge(df_all, df_attr, how='left', on='product_uid')

def replace_nan(s):
        if pd.isnull(s)==True:
                s=""
        return s

p = df_all.keys()
for i in range(len(p)):
    print p[i]






df_all['search_term_stemmed'] = df_all['search_term_stemmed'].map(lambda x:replace_nan(x))
df_all['product_title_stemmed'] = df_all['product_title_stemmed'].map(lambda x:replace_nan(x))
df_all['product_description_stemmed'] = df_all['product_description_stemmed'].map(lambda x:replace_nan(x))
df_all['brand_parsed'] = df_all['brand_parsed'].map(lambda x:replace_nan(x))
df_all['material_parsed'] = df_all['material_parsed'].map(lambda x:replace_nan(x))
df_all['attribute_bullets_stemmed'] = df_all['attribute_bullets_stemmed'].map(lambda x:replace_nan(x))
df_all['value'] = df_all['value'].map(lambda x:replace_nan(x))

df_all['search_term'] = df_all['search_term'].map(lambda x:replace_nan(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:replace_nan(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:replace_nan(x))
df_all['brand'] = df_all['brand'].map(lambda x:replace_nan(x))
df_all['material'] = df_all['material'].map(lambda x:replace_nan(x))
df_all['attribute_bullets'] = df_all['attribute_bullets'].map(lambda x:replace_nan(x))
df_all['value'] = df_all['value'].map(lambda x:replace_nan(x))



st = df_all["search_term_stemmed"]
pt = df_all["product_title_stemmed"]
pd = df_all["product_description_stemmed"]
br = df_all["brand_parsed"]
mr = df_all["material_parsed"]
ab = df_all["attribute_bullets_stemmed"]
at = df_all["value"]


##st + pt +pd vocab
#t = list()
#for i in range(len(st)):
#    p = st[i].split()
#    t.append(p)
#   
#for i in range(len(pt)):
#    p = pt[i].split()
#    t.append(p)
#    
#for i in range(len(pd)):
#    p = pd[i].split()
#    t.append(p)
#     
##for i in range(len(br)):
##    p = br[i].split()
##    t.append(p)
##
##for i in range(len(mr)):
##    p = mr[i].split()
##    t.append(p)
#    
#for i in range(len(ab)):
#    p = ab[i].split()
#    t.append(p)
#    
#for i in range(len(at)):
#    p = at[i].split()
#    t.append(p)
    
print "first vocab"
#st conc pt conc pd vocab
t1 = list()
for i in range(len(st)):
    p = st[i].split()+pt[i].split()+pd[i].split()+br[i].split()+mr[i].split()+ab[i].split()+at[i].split()
    t1.append(p)

print "second vocab"

#st + pt +pd +br + mr vocab w/o pars
st1 = df_all["search_term"]
pt1 = df_all["product_title"]
pd1 = df_all["product_description"]
br1 = df_all["brand"]
mr1 = df_all["material"]
ab1 = df_all["attribute_bullets"]
at1 = df_all["value"]

#t2 = list()
#for i in range(len(st)):
#    p = st1[i].split()
#    t2.append(p)
#   
#for i in range(len(pt)):
#    p = pt1[i].split()
#    t2.append(p)
# 
#for i in range(len(pd)):
#    p = pd1[i].split()
#    t2.append(p)
#
##for i in range(len(br)):
##    p = br1[i].split()
##    t2.append(p)
## 
##for i in range(len(mr)):
##    p = mr1[i].split()
##    t2.append(p)
#
#for i in range(len(ab1)):
#    p = ab1[i].split()
#    t2.append(p)
# 
#for i in range(len(at1)):
#    p = at1[i].split()
#    t2.append(p) 
# 
#print "third vocab"   

#st conc pt conc pd conc br conc mr vocab w/o pars
t3 = list()
for i in range(len(st)):
    p = st1[i].split()+pt1[i].split()+pd1[i].split()+br1[i].split()+mr1[i].split()+ab1[i].split()+at1[i].split()
    t3.append(p)

print "fourth vocab" 


#model0 = gensim.models.Word2Vec(t, sg=1, window=10, sample=1e-5, negative=5, size=300)
model1 = gensim.models.Word2Vec(t1, sg=1, window=10, sample=1e-5, negative=5, size=300)
print "model prepared"
#model2 = gensim.models.Word2Vec(t2, sg=1, window=10, sample=1e-5, negative=5, size=300)
model3 = gensim.models.Word2Vec(t3, sg=1, window=10, sample=1e-5, negative=5, size=300)
print "model prepared"
#model4 = gensim.models.Word2Vec(t, sg=0,  hs=1, window=10,   size=300)
#model5 = gensim.models.Word2Vec(t1, sg=0, hs=1,window=10,   size=300)
#model6 = gensim.models.Word2Vec(t2, sg=0, hs=1, window=10,   size=300)
#model7 = gensim.models.Word2Vec(t3, sg=0, hs=1,window=10,   size=300)



#model_list=[model0,model1,model2,model3]   #,model4  ,model5,model6,model7]
model_list=[model1,model3]
n_sim=list()

for model in model_list:
    print "model features calculation"
    n_sim_pt=list()
    for i in range(len(st)):
        w1=st[i].split()
        w2=pt[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_pt.append(0)
        else:    
            n_sim_pt.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_pt)
    
    n_sim_pd=list()
    for i in range(len(st)):
        w1=st[i].split()
        w2=pd[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_pd.append(0)
        else:    
            n_sim_pd.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_pd)

    n_sim_at=list()
    for i in range(len(st)):
        w1=st[i].split()
        w2=at[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_at.append(0)
        else:    
            n_sim_at.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_at)

    n_sim_all=list()
    for i in range(len(st)):
        w1=st[i].split()
        w2=pt[i].split()+pd[i].split()+br[i].split()+mr[i].split()+ab[i].split()+at[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_all.append(0)
        else:    
            n_sim_all.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_all)

    n_sim_all1=list()
    for i in range(len(st)):
        w1=st1[i].split()
        w2=pt1[i].split()+pd1[i].split()+br1[i].split()+mr1[i].split()+ab1[i].split()+at1[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_all1.append(0)
        else:    
            n_sim_all1.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_all1)
    
    n_sim_ptpd=list()
    for i in range(len(st)):
        w1=pt[i].split()
        w2=pd[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_ptpd.append(0)
        else:    
            n_sim_ptpd.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_ptpd)
    print "model features done"

st_names=["id"]    
#for j in range(len(n_sim)):   
name_list=list([6,7,8,9,10,11,18,19,20,21,22,23])
for j in range(len(n_sim)):   

    df_all["word2vec_"+str(name_list[j])]=n_sim[j]
    st_names.append("word2vec_"+str(name_list[j]))
    


b=df_all[st_names]
b.to_csv(FEATURES_DIR+"/df_word2vec_wo_google_dict.csv", index=False) 




