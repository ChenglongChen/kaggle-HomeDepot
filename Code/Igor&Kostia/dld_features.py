# -*- coding: utf-8 -*-
"""
Code for calculating some count features using Damerau–Levenshtein distance.
Competition: HomeDepot Search Relevance
Author: Kostia Omelianchuk
Team: Turing test
"""

from config_IgorKostia import *

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
import nltk
from time import time
import re
import os
import math as m



#loading data
df_all=pd.read_csv(PROCESSINGTEXT_DIR+"/df_train_and_test_processed.csv", encoding="ISO-8859-1")
df_all1=pd.read_csv(PROCESSINGTEXT_DIR+"/df_product_descriptions_processed.csv", encoding="ISO-8859-1")
df_all2 = pd.merge(df_all, df_all1, how="left", on="product_uid")
df_all = df_all2
df_all1=pd.read_csv(PROCESSINGTEXT_DIR+"/df_attribute_bullets_processed.csv", encoding="ISO-8859-1")
df_all2 = pd.merge(df_all, df_all1, how="left", on="product_uid")
df_all = df_all2
df_attr = pd.read_csv(PROCESSINGTEXT_DIR+'/df_attributes_kostia.csv', encoding="ISO-8859-1")
df_all = pd.merge(df_all, df_attr, how='left', on='product_uid')



#replace nan
for var in df_all.keys():
    df_all[var]=df_all[var].fillna("")

p = df_all.keys()
for i in range(len(p)):
    print p[i]

def replace_nan(s):
        if pd.isnull(s)==True:
                s=""
        return s



df_all['search_term_stemmed'] = df_all['search_term_stemmed'].map(lambda x:replace_nan(x))
df_all['product_title_stemmed'] = df_all['product_title_stemmed'].map(lambda x:replace_nan(x))
df_all['product_description_stemmed'] = df_all['product_description_stemmed'].map(lambda x:replace_nan(x))
df_all['brand_parsed'] = df_all['brand_parsed'].map(lambda x:replace_nan(x))
df_all['material_parsed'] = df_all['material_parsed'].map(lambda x:replace_nan(x))
df_all['attribute_bullets_stemmed'] = df_all['attribute_bullets_stemmed'].map(lambda x:replace_nan(x))
df_all['attribute_stemmed'] = df_all['attribute_stemmed'].map(lambda x:replace_nan(x))

df_all['search_term'] = df_all['search_term'].map(lambda x:replace_nan(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:replace_nan(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:replace_nan(x))
df_all['brand'] = df_all['brand'].map(lambda x:replace_nan(x))
df_all['material'] = df_all['material'].map(lambda x:replace_nan(x))
df_all['attribute_bullets'] = df_all['attribute_bullets'].map(lambda x:replace_nan(x))
df_all['value'] = df_all['value'].map(lambda x:replace_nan(x))


"""
st = df_all["search_term_stemmed"]
pt = df_all["product_title_stemmed"]
pd = df_all["product_description_stemmed"]
br = df_all["brand_parsed"]
mr = df_all["material_parsed"]
ab = df_all["attribute_bullets_stemmed"]
at = df_all["attribute_stemmed"]
"""

#make one big string
df_all['product_info'] = df_all['search_term']+"\t"+df_all['search_term_stemmed']+ "\t"  \
             +df_all['product_title']+"\t"+df_all['product_title_stemmed'] + "\t"\
             +df_all['product_description'] + "\t"+df_all['product_description_stemmed'] + "\t"\
             +df_all['attribute_bullets']+ "\t"+df_all['attribute_bullets_stemmed'] + "\t"\
             +df_all['value']+ "\t"+df_all['attribute_stemmed']
             
df_all['product_info'] = df_all['product_info'].map(lambda x:replace_nan(x))


#dld functions
def dld(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in xrange(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in xrange(-1,lenstr2+1):
        d[(-1,j)] = j+1
 
    for i in xrange(lenstr1):
        for j in xrange(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
 
    return d[lenstr1-1,lenstr2-1]



def str_common_word_dld(str1, str2):
      m = 0 
      for word in str1.split(): 
           if len(word)>1 :
               for word2 in str2.split():              
                   if dld(word,word2)<min(3,max(len(word)-3,1)): 
                            m = m + 1
      return m


def str_common_word_string_only_dld(str1, str2):
     m = 0 
     for word in str1.split():          
             if len(word)>1 and len(re.findall(r'\d+', word))==0 :
                  for word2 in str2.split():        
                      if dld(word,word2)<min(3,max(len(word)-3,1)): 
                            m = m + 1
     return m



 
def str_2common_words_dld(str1, str2):
        found=0
        found_string_only=0
        words_in_query=str1.split()
        words_in_text=str2.split()
        #print words_in_query
        for cnt in range(0,len(words_in_query)-1):
            two_words='_'+words_in_query[cnt]+' '+words_in_query[cnt+1]+' '
            if len(words_in_query[cnt])>1 and len(words_in_query[cnt+1])>1:
                    for cnt2 in range(0,len(words_in_text)-1):
                        two_words_text ='_'+words_in_text[cnt2]+' '+words_in_text[cnt2+1]+' '  
                        if dld(two_words,two_words_text)<min(3,max(len(two_words_text)-3,1)): 
                          found = found + 1
                        #       if len(re.findall(r'\d+', two_words))==0:
                        #        found_string_only+=1
        return found
       


def str_2common_words_string_only_dld(str1, str2):
        found=0
        found_string_only=0
        words_in_query=str1.split()
        words_in_text=str2.split()
        #print words_in_query
        for cnt in range(0,len(words_in_query)-1):
            two_words='_'+words_in_query[cnt]+' '+words_in_query[cnt+1]+' '
            if len(words_in_query[cnt])>1 and len(words_in_query[cnt+1])>1 and len(re.findall(r'\d+', two_words))==0:
                    for cnt2 in range(0,len(words_in_text)-1):
                        two_words_text ='_'+words_in_text[cnt2]+' '+words_in_text[cnt2+1]+' '  
                        if dld(two_words,two_words_text)<min(3,max(len(two_words_text)-3,1)): 
                         found_string_only = found_string_only + 1
                        #       if len(re.findall(r'\d+', two_words))==0:
                        #        found_string_only+=1
        return found_string_only
        



def str_common_digits_1(str1, str2):
        found=0
        found_words_only=0
        digits_in_query=list(set(re.findall(r'\d+\/\d+|\d+\.\d+|\d+', str1)))
        digits_in_text=re.findall(r'\d+\/\d+|\d+\.\d+|\d+', str2)
        for digit in digits_in_query:
                found = found + digits_in_text.count(digit)
                       
        return found


def add_one(x):
        if x==0:
                x=1
        return x

first_list=["st","pt","pd","ab","at"]  
first_num=[0,2,4,6,8]
second_list=["sts","pts","pds","abs","ats"]  
second_num=[1,3,5,7,9]       

len_of_features=len(df_all.keys())
idf=df_all["id"]        

#calculate dld features
for i in range(0,5):
    t0 = time()
    df_all['1word_dld_in_'+(first_list[i])] = df_all['product_info'].map(lambda x:str_common_word_dld(x.split('\t')[0],x.split('\t')[first_num[i]]))
    df_all['1word_dld_in_'+(second_list[i])] = df_all['product_info'].map(lambda x:str_common_word_dld(x.split('\t')[1],x.split('\t')[second_num[i]]))
    print '1word_dld time:',round(time()-t0,3) ,'s\n'
    
    t0 = time()
    df_all['1word_string_dld_in_'+(first_list[i])] = df_all['product_info'].map(lambda x:str_common_word_string_only_dld(x.split('\t')[0],x.split('\t')[first_num[i]]))
    df_all['1word_string_dld_in_'+(second_list[i])] = df_all['product_info'].map(lambda x:str_common_word_string_only_dld(x.split('\t')[1],x.split('\t')[second_num[i]]))
    print '1word_string_dld time:',round(time()-t0,3) ,'s\n'

    t0 = time()
    df_all['2word_dld_in_'+(first_list[i])] = df_all['product_info'].map(lambda x:str_2common_words_dld(x.split('\t')[0],x.split('\t')[first_num[i]]))
    df_all['2word_dld_in_'+(second_list[i])] = df_all['product_info'].map(lambda x:str_2common_words_dld(x.split('\t')[1],x.split('\t')[second_num[i]]))
    print '2word_dld time:',round(time()-t0,3) ,'s\n'

    t0 = time()
    df_all['2word_string_dld_in_'+(first_list[i])] = df_all['product_info'].map(lambda x:str_2common_words_string_only_dld(x.split('\t')[0],x.split('\t')[first_num[i]]))
    df_all['2word_string_dld_in_'+(second_list[i])] = df_all['product_info'].map(lambda x:str_2common_words_string_only_dld(x.split('\t')[1],x.split('\t')[second_num[i]]))
    print '2word_string_dld time:',round(time()-t0,3) ,'s\n'    


#save result
st_names=list(df_all.keys()[len_of_features:])
st_names.append("id")
b=df_all[st_names]
b.to_csv(FEATURES_DIR+"/dld_features.csv", index=False)






