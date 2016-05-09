# -*- coding: utf-8 -*-
"""
Generate features.

This file is the same as feature_extraction1.py, except the input 
for the file is generated without the Google dict from the forum.

Competition: HomeDepot Search Relevance
Author: Igor Buinyi
Team: Turing test
"""

from config_IgorKostia import *

import numpy as np
import pandas as pd
from time import time
import re
import csv
import os
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
stoplist.append('till')
stoplist_wo_can=stoplist[:]
stoplist_wo_can.remove('can')

from homedepot_functions import *


t0 = time()
t1 = time()

#################################################################
### STEP 0: Load the results of text preprocessing
#################################################################

df_pro_desc = pd.read_csv(PROCESSINGTEXT_DIR+'/df_product_descriptions_processed_wo_google.csv')
df_attr_bullets = pd.read_csv(PROCESSINGTEXT_DIR+'/df_attribute_bullets_processed_wo_google.csv')
df_all = pd.read_csv(PROCESSINGTEXT_DIR+'/df_train_and_test_processed_wo_google.csv')
print 'loading time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()



df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_attr_bullets, how='left', on='product_uid')
print 'merging time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()



for var in df_all.keys():
    df_all[var]=df_all[var].fillna("") 




### the function returns text after a specific word
### for example, extract_after_word('faucets for kitchen') 
### will return 'kitchen'
def extract_after_word(s,word):
    output=""
    if word in s:
        srch= re.search(r'(?<=\b'+word+'\ )[a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]+',s)
        if srch!=None:
            output=srch.group(0)
    return output

### get 'for' and 'with' parts from query and 'without' from product title
df_all['search_term_for']=df_all['search_term_parsed'].map(lambda x: extract_after_word(x,'for'))
df_all['search_term_for_stemmed']=df_all['search_term_for'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))

df_all['search_term_with']=df_all['search_term_parsed'].map(lambda x: extract_after_word(x,'with'))
df_all['search_term_with_stemmed']=df_all['search_term_with'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))

df_all['product_title_parsed_without']=df_all['product_title_parsed'].map(lambda x: extract_after_word(x,'without'))
df_all['product_title_without_stemmed']=df_all['product_title_parsed_without'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))



### save the list of string variables which are not features
string_variables_list=list(df_all.keys())
print str(len(string_variables_list) ) + " total variables..."
string_variables_list.remove('id')
string_variables_list.remove('product_uid')
string_variables_list.remove('relevance')
string_variables_list.remove('is_query_misspelled')
print "including "+ str(len(string_variables_list) ) + " text variables to drop later"



t0 = time()
#################################################################
### STEP 1: Dummy for no attributes and empty attribute bullets
#################################################################
df_attr_bullets['has_attributes_dummy']=1
df_all = pd.merge(df_all, df_attr_bullets[['product_uid','has_attributes_dummy']], how='left', on='product_uid')
df_all['has_attributes_dummy']= df_all['has_attributes_dummy'].fillna(0)


df_all['no_bullets_dummy'] = df_all['attribute_bullets'].map(lambda x:int(len(x)==0))
from google_dict import *
df_all['is_replaced_using_google_dict']=df_all['search_term'].map(lambda x: 1 if x in google_dict.keys() else 0)



df_attr_bullets=df_attr_bullets.drop(list(df_attr_bullets.keys()),axis=1)
df_pro_desc=df_pro_desc.drop(list(df_pro_desc.keys()),axis=1)


#################################################################
### STEP 2: Basic text features
#################################################################

### Calculated here are basic text features:
### * number of words/digits
### * average length of word
### * ratio of vowels in query
### * average length of word
### * number/length of brands/materials
### * percentage of digits


def sentence_statistics(s):
    s= re.sub('[^a-zA-Z0-9\ \%\$\-]', '', s)
    word_list=s.split()
    meaningful_word_list=[word for word in s.split() if len(re.findall(r'\d+', word))==0 and len(wn.synsets(word))>0]
    vowels=sum([len(re.sub('[^aeiou]', '', word)) for word in word_list])
    letters = sum([len(word) for word in word_list])
    
    return len(word_list), len(meaningful_word_list), 1.0*sum([len(word) for word in word_list])/len(word_list), 1.0*vowels/letters

df_all['query_sentence_stats_tuple'] = df_all['search_term_parsed'].map(lambda x:  sentence_statistics(x) )
df_all['len_of_meaningful_words_in_query'] = df_all['query_sentence_stats_tuple'].map(lambda x:  x[1] )
df_all['ratio_of_meaningful_words_in_query'] = df_all['query_sentence_stats_tuple'].map(lambda x:  1.0*x[1]/x[0] )
df_all['avg_wordlength_in_query'] = df_all['query_sentence_stats_tuple'].map(lambda x:  x[2] )
df_all['ratio_vowels_in_query'] = df_all['query_sentence_stats_tuple'].map(lambda x:  x[3] )
df_all=df_all.drop(['query_sentence_stats_tuple'],axis=1)


df_all['initial_len_of_query'] = df_all['search_term_stemmed'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_query_woBM'] = df_all['search_term_stemmed_woBM'].map(lambda x:len(x.split())).astype(np.int64)

df_all['len_of_query_for'] = df_all['search_term_for_stemmed'].map(lambda x:len(words_wo_digits(x, minLength=1).split())).astype(np.int64)
df_all['len_of_query_with'] = df_all['search_term_with_stemmed'].map(lambda x:len(words_wo_digits(x, minLength=1).split())).astype(np.int64)
df_all['len_of_prtitle_without'] = df_all['product_title_without_stemmed'].map(lambda x:len(words_wo_digits(x, minLength=1).split())).astype(np.int64)

df_all['len_of_query_string_only_woBM'] = df_all['search_term_stemmed_woBM'].map(lambda x:len(words_wo_digits(x, minLength=1).split())).astype(np.int64)

df_all['len_of_query_w_dash_woBM'] = df_all['search_term_stemmed_woBM'].map(lambda x:len(words_w_dash(x).split())).astype(np.int64)
df_all['len_of_title_w_dash_woBM'] = df_all['product_title_stemmed_woBM'].map(lambda x:len(words_w_dash(x).split())).astype(np.int64)

df_all['len_of_product_title_woBM'] = df_all['product_title_stemmed_woBM'].map(lambda x:len(x.split())).astype(np.int64)

df_all['len_of_product_description_woBM'] = df_all['product_description_stemmed_woBM'].map(lambda x:len(x.split())).astype(np.int64)

df_all['len_of_attribute_bullets_woBM'] = df_all['attribute_bullets_stemmed_woBM'].map(lambda x:len(x.split())).astype(np.int64)

df_all['len_of_brands_in_query'] = df_all['brands_in_search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_brands_in_query'] = df_all['brands_in_search_term'].map(lambda x:len(x.split(";"))).astype(np.int64)
df_all['len_of_materials_in_query'] = df_all['materials_in_search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_materials_in_query'] = df_all['materials_in_search_term'].map(lambda x:len(x.split(";"))).astype(np.int64)

df_all['len_of_brands_in_product_title'] = df_all['brands_in_product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_brands_in_product_title'] = df_all['brands_in_product_title'].map(lambda x:len(x.split(";"))).astype(np.int64)
df_all['len_of_materials_in_product_title'] = df_all['materials_in_product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_materials_in_product_title'] = df_all['materials_in_product_title'].map(lambda x:len(x.split(";"))).astype(np.int64)

df_all['len_of_brands_in_product_description'] = df_all['brands_in_product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_brands_in_product_description'] = df_all['brands_in_product_description'].map(lambda x:len(x.split(";"))).astype(np.int64)
df_all['len_of_materials_in_product_description'] = df_all['materials_in_product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_materials_in_product_description'] = df_all['materials_in_product_description'].map(lambda x:len(x.split(";"))).astype(np.int64)

df_all['len_of_brands_in_attribute_bullets'] = df_all['brands_in_attribute_bullets'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_brands_in_attribute_bullets'] = df_all['brands_in_attribute_bullets'].map(lambda x:len(x.split(";"))).astype(np.int64)
df_all['len_of_materials_in_attribute_bullets'] = df_all['materials_in_attribute_bullets'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_materials_in_attribute_bullets'] = df_all['materials_in_attribute_bullets'].map(lambda x:len(x.split(";"))).astype(np.int64)

df_all['len_of_query']=df_all['len_of_query_woBM']+df_all['size_of_brands_in_query']+df_all['size_of_materials_in_query']
df_all['len_of_product_title']=df_all['len_of_product_title_woBM']+df_all['size_of_brands_in_product_title']

df_all['len_of_product_description']=df_all['len_of_product_title_woBM']+df_all['size_of_brands_in_product_description']+df_all['size_of_materials_in_product_description']
### The previous line is incorrect.
### It should be 
### df_all['len_of_product_description']=df_all['len_of_product_description_woBM']+df_all['size_of_brands_in_product_description']+df_all['size_of_materials_in_product_description']

df_all['len_of_attribute_bullets']=df_all['len_of_attribute_bullets_woBM']+df_all['size_of_brands_in_attribute_bullets']+df_all['size_of_materials_in_attribute_bullets']


df_all['len_of_query_keys'] = df_all['search_term_keys_stemmed'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_product_title_keys'] = df_all['product_title_keys_stemmed'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_query_thekey'] = df_all['search_term_thekey'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_product_title_thekey'] = df_all['product_title_thekey'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_query_beforethekey'] = df_all['search_term_beforethekey'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_product_title_beforethekey'] = df_all['product_title_beforethekey'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_query_before2thekey'] = df_all['search_term_before2thekey'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_product_title_before2thekey'] = df_all['product_title_before2thekey'].map(lambda x:len(x.split())).astype(np.int64)


def find_ratio(a,b):
    if b==0:
        return 0
    else:
        return min(1,1.0*a/b)
        
for column_name in ['search_term', 'product_title', 'product_description','attribute_bullets']:
    df_all['ratio_of_nn_important_in_'+column_name]= df_all.apply(lambda x: \
        find_ratio(len(nn_important_words(x['search_term_tokens']).split()), len(x[column_name+'_parsed'])) ,axis=1)
    df_all['ratio_of_nn_unimportant_in_'+column_name]= df_all.apply(lambda x: \
        find_ratio(len(nn_unimportant_words(x['search_term_tokens']).split()), len(x[column_name+'_parsed'])) ,axis=1)
    df_all['ratio_of_vb_in_'+column_name]= df_all.apply(lambda x: \
        find_ratio(len(vb_words(x['search_term_tokens']).split()), len(x[column_name+'_parsed'])) ,axis=1)
    df_all['ratio_of_vbg_in_'+column_name]= df_all.apply(lambda x: \
        find_ratio(len(vbg_words(x['search_term_tokens']).split()), len(x[column_name+'_parsed'])) ,axis=1)
    df_all['ratio_of_jj_rb_in_'+column_name]= df_all.apply(lambda x: \
        find_ratio(len(nn_unimportant_words(x['search_term_tokens']).split()), len(x[column_name+'_parsed'])) ,axis=1)




print 'len_of_something time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


def perc_digits_in_str(s):
    if len(s.split())==0:
        output=0
    else:
        output =1.0 -1.0*len(words_wo_digits(s, minLength=1).split()) / len(s.split())
    return output

df_all['perc_digits_in_query'] = df_all['search_term_stemmed'].map(lambda x: perc_digits_in_str(x)  )
df_all['perc_digits_in_title'] = df_all['product_title_stemmed'].map(lambda x: perc_digits_in_str(x)  )
df_all['perc_digits_in_description'] = df_all['product_description_stemmed'].map(lambda x: perc_digits_in_str(x) )
df_all['perc_digits_in_bullets'] = df_all['attribute_bullets_stemmed'].map(lambda x: perc_digits_in_str(x) )

print 'perc_digits time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


#################################################################
### STEP 3: Measure whether brand and materials in query match other text
#################################################################

### the following function compares query with the corresponding product title,
### prodcut description or attribute bullets and returns:
### * number of brands/materials fully matched
### * number of brands/materials partially matched ('behr' and 'behr premium')
### * number of brands/materials with assumed match (first word of brand found, e.g. 'handy home products' and 'handy paint pail')
### * number of brands/materials in query not matched
### * convoluted output: 3 if all brands/materials fully matched,
###                      2 if all matched at least partially
###                      1 if at least one is matched but at least one is not matched
###                      0 if no brand/material in query
###                     -1 if there is brand/material in query but no brand/material in text
###                     -2 if there are brands different brands/materials in query and text 

def query_brand_material_in_attribute(str_query_brands,str_attribute_brands):
    list_query_brands=list(set(str_query_brands.split(";")))
    list_attribute_brands=list(set(str_attribute_brands.split(";")))
    while '' in list_query_brands:
        list_query_brands.remove('')
    while '' in list_attribute_brands:
        list_attribute_brands.remove('')
    
    str_attribute_brands=" ".join(str_attribute_brands.split(";"))    
    full_match=0
    partial_match=0
    assumed_match=0
    no_match=0
    num_of_query_brands=len(list_query_brands)
    num_of_attribute_brands=len(list_attribute_brands)
    if num_of_query_brands>0:
        for brand in list_query_brands:
            if brand in list_attribute_brands:
                full_match+=1
            elif ' '+brand+' ' in ' '+str_attribute_brands+' ':
                partial_match+=1
            elif (' '+brand.split()[0] in ' '+str_attribute_brands and brand.split()[0][0] not in "0123456789") or \
            (len(brand.split())>1 and (' '+brand.split()[0]+' '+brand.split()[1]) in ' '+str_attribute_brands):
                assumed_match+=1
            else:
                no_match+=1
                
    convoluted_output=0 # no brand in query
    if num_of_query_brands>0:
        if num_of_attribute_brands==0:
            convoluted_output = -1 # no brand in text, but there is brand in query
        elif no_match==0:
            if assumed_match==0:
                convoluted_output=3 # all brands fully matched
            else:
                convoluted_output=2 # all brands matched at least partially
        else:
            if full_match+ partial_match+ assumed_match>0:
                convoluted_output = 1 # one brand matched but the other is not
            else:
                convoluted_output= -2  #brand mismatched
    
    return full_match, partial_match, assumed_match, no_match, convoluted_output


df_all['brands_all']=df_all['brands_in_search_term']+"\t"+df_all['brands_in_product_title']+"\t"+df_all['brands_in_product_description']+"\t"\
+df_all['brands_in_attribute_bullets']+"\t"+df_all['brand_parsed']

df_all['query_brand_in_brand_tuple']=df_all['brands_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[4]))
df_all['query_brand_in_brand_fullmatch']=df_all['query_brand_in_brand_tuple'].map(lambda x: x[0])
df_all['query_brand_in_brand_partialmatch']=df_all['query_brand_in_brand_tuple'].map(lambda x: x[1])
df_all['query_brand_in_brand_assumedmatch']=df_all['query_brand_in_brand_tuple'].map(lambda x: x[2])
df_all['query_brand_in_brand_nomatch']=df_all['query_brand_in_brand_tuple'].map(lambda x: x[3])
df_all['query_brand_in_brand_convoluted']=df_all['query_brand_in_brand_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_brand_in_brand_tuple'],axis=1)

df_all['query_brand_in_all_tuple']=df_all['brands_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],";".join(x.split("\t")[1:])))
df_all['query_brand_in_all_fullmatch']=df_all['query_brand_in_all_tuple'].map(lambda x: x[0])
df_all['query_brand_in_all_partialmatch']=df_all['query_brand_in_all_tuple'].map(lambda x: x[1])
df_all['query_brand_in_all_assumedmatch']=df_all['query_brand_in_all_tuple'].map(lambda x: x[2])
df_all['query_brand_in_all_nomatch']=df_all['query_brand_in_all_tuple'].map(lambda x: x[3])
df_all['query_brand_in_all_convoluted']=df_all['query_brand_in_all_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_brand_in_all_tuple'],axis=1)

df_all['query_brand_in_title_tuple']=df_all['brands_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[1]))
df_all['query_brand_in_title_fullmatch']=df_all['query_brand_in_title_tuple'].map(lambda x: x[0])
df_all['query_brand_in_title_partialmatch']=df_all['query_brand_in_title_tuple'].map(lambda x: x[1])
df_all['query_brand_in_title_assumedmatch']=df_all['query_brand_in_title_tuple'].map(lambda x: x[2])
df_all['query_brand_in_title_nomatch']=df_all['query_brand_in_title_tuple'].map(lambda x: x[3])
df_all['query_brand_in_title_convoluted']=df_all['query_brand_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_brand_in_title_tuple'],axis=1)

df_all['query_brand_in_description_tuple']=df_all['brands_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[2]))
df_all['query_brand_in_description_convoluted']=df_all['query_brand_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_brand_in_description_tuple'],axis=1)

df_all['query_brand_in_bullets_tuple']=df_all['brands_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[3]))
df_all['query_brand_in_bullets_convoluted']=df_all['query_brand_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_brand_in_bullets_tuple'],axis=1)

df_all=df_all.drop(['brands_all'],axis=1)

print 'create brand match variables time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()



########################################

df_all['materials_all']=df_all['materials_in_search_term']+"\t"+df_all['materials_in_product_title']+"\t"+df_all['materials_in_product_description']+"\t"\
+df_all['materials_in_attribute_bullets']+"\t"+df_all['material_parsed']

df_all['query_material_in_material_tuple']=df_all['materials_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[4]))
df_all['query_material_in_material_fullmatch']=df_all['query_material_in_material_tuple'].map(lambda x: x[0])
df_all['query_material_in_material_partialmatch']=df_all['query_material_in_material_tuple'].map(lambda x: x[1])
df_all['query_material_in_material_assumedmatch']=df_all['query_material_in_material_tuple'].map(lambda x: x[2])
df_all['query_material_in_material_nomatch']=df_all['query_material_in_material_tuple'].map(lambda x: x[3])
df_all['query_material_in_material_convoluted']=df_all['query_material_in_material_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_material_in_material_tuple'],axis=1)

df_all['query_material_in_all_tuple']=df_all['materials_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],";".join(x.split("\t")[1:])))
df_all['query_material_in_all_fullmatch']=df_all['query_material_in_all_tuple'].map(lambda x: x[0])
df_all['query_material_in_all_partialmatch']=df_all['query_material_in_all_tuple'].map(lambda x: x[1])
df_all['query_material_in_all_assumedmatch']=df_all['query_material_in_all_tuple'].map(lambda x: x[2])
df_all['query_material_in_all_nomatch']=df_all['query_material_in_all_tuple'].map(lambda x: x[3])
df_all['query_material_in_all_convoluted']=df_all['query_material_in_all_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_material_in_all_tuple'],axis=1)

df_all['query_material_in_title_tuple']=df_all['materials_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[1]))
df_all['query_material_in_title_convoluted']=df_all['query_material_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_material_in_title_tuple'],axis=1)

df_all['query_material_in_description_tuple']=df_all['materials_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[2]))
df_all['query_material_in_description_convoluted']=df_all['query_material_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_material_in_description_tuple'],axis=1)

df_all['query_material_in_bullets_tuple']=df_all['materials_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[3]))
df_all['query_material_in_bullets_convoluted']=df_all['query_material_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_material_in_bullets_tuple'],axis=1)

df_all=df_all.drop(['materials_all'],axis=1)

print 'create material match variables time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()





#################################################################
### STEP 4: Count number of similar words/letters/bigrams etc
#################################################################

################################
### query vs product title


df_all['wordFor_in_title_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_for_stemmed'],x['product_title_stemmed'],string_only=True),axis=1)
df_all['wordFor_in_title_string_only_num'] = df_all['wordFor_in_title_string_only_tuple'].map(lambda x: x[0])
df_all['wordFor_in_title_string_only_let'] = df_all['wordFor_in_title_string_only_tuple'].map(lambda x: x[2])
df_all['wordFor_in_title_string_only_letratio'] = df_all['wordFor_in_title_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['wordFor_in_title_string_only_tuple'],axis=1)

df_all['wordWith_in_title_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_with_stemmed'],x['product_title_stemmed'],string_only=True),axis=1)
df_all['wordWith_in_title_string_only_num'] = df_all['wordWith_in_title_string_only_tuple'].map(lambda x: x[0])
df_all['wordWith_in_title_string_only_let'] = df_all['wordWith_in_title_string_only_tuple'].map(lambda x: x[2])
df_all['wordWith_in_title_string_only_letratio'] = df_all['wordWith_in_title_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['wordWith_in_title_string_only_tuple'],axis=1)

df_all['prtitleWithout_in_query_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['product_title_without_stemmed'],x['search_term_stemmed'],string_only=True),axis=1)
df_all['prtitleWithout_in_query_string_only_num'] = df_all['prtitleWithout_in_query_string_only_tuple'].map(lambda x: x[0])
df_all['prtitleWithout_in_query_string_only_let'] = df_all['prtitleWithout_in_query_string_only_tuple'].map(lambda x: x[2])
df_all['prtitleWithout_in_query_string_only_letratio'] = df_all['prtitleWithout_in_query_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['prtitleWithout_in_query_string_only_tuple'],axis=1)

df_all['query_in_title']=df_all.apply(lambda x: \
            query_in_text(x['search_term_stemmed'],x['product_title_stemmed']),axis=1)

df_all['word_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_stemmed'],x['product_title_stemmed']),axis=1)
df_all['word_in_title_num'] = df_all['word_in_title_tuple'].map(lambda x: x[0])
df_all['word_in_title_sum'] = df_all['word_in_title_tuple'].map(lambda x: x[1])
df_all['word_in_title_let'] = df_all['word_in_title_tuple'].map(lambda x: x[2])
df_all['word_in_title_numratio'] = df_all['word_in_title_tuple'].map(lambda x: x[3])
df_all['word_in_title_letratio'] = df_all['word_in_title_tuple'].map(lambda x: x[4])
df_all['word_in_title_string'] = df_all['word_in_title_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['word_in_title_tuple'],axis=1)

df_all['word_in_title_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_stemmed'],x['product_title_stemmed'],string_only=True),axis=1)
df_all['word_in_title_string_only_num'] = df_all['word_in_title_string_only_tuple'].map(lambda x: x[0])
df_all['word_in_title_string_only_sum'] = df_all['word_in_title_string_only_tuple'].map(lambda x: x[1])
df_all['word_in_title_string_only_let'] = df_all['word_in_title_string_only_tuple'].map(lambda x: x[2])
df_all['word_in_title_string_only_numratio'] = df_all['word_in_title_string_only_tuple'].map(lambda x: x[3])
df_all['word_in_title_string_only_letratio'] = df_all['word_in_title_string_only_tuple'].map(lambda x: x[4])
df_all['word_in_title_string_only_string'] = df_all['word_in_title_string_only_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['word_in_title_string_only_tuple'],axis=1)

df_all['word_in_title_w_dash']=df_all.apply(lambda x: \
            str_common_word(words_w_dash(x['search_term_stemmed']),words_w_dash(x['product_title_stemmed']))[0],axis=1)

df_all['two_words_in_title_tuple']=df_all.apply(lambda x: \
            str_2common_words(x['search_term_stemmed'],x['product_title_stemmed']),axis=1)
df_all['two_words_in_title_num'] = df_all['two_words_in_title_tuple'].map(lambda x: x[0])
df_all['two_words_in_title_sum'] = df_all['two_words_in_title_tuple'].map(lambda x: x[1])
df_all['two_words_in_title_let'] = df_all['two_words_in_title_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['two_words_in_title_tuple'],axis=1)

df_all['two_words_in_title_string_only_tuple']=df_all.apply(lambda x: \
            str_2common_words(x['search_term_stemmed'],x['product_title_stemmed'],string_only=True),axis=1)
df_all['two_words_in_title_string_only_num'] = df_all['two_words_in_title_string_only_tuple'].map(lambda x: x[0])
df_all['two_words_in_title_string_only_sum'] = df_all['two_words_in_title_string_only_tuple'].map(lambda x: x[1])
df_all['two_words_in_title_string_only_let'] = df_all['two_words_in_title_string_only_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['two_words_in_title_string_only_tuple'],axis=1)

df_all['common_digits_in_title_tuple']=df_all.apply(lambda x: \
            str_common_digits(x['search_term_stemmed'],x['product_title_stemmed']),axis=1)
df_all['len_of_digits_in_query'] = df_all['common_digits_in_title_tuple'].map(lambda x: x[0])
df_all['len_of_digits_in_title'] = df_all['common_digits_in_title_tuple'].map(lambda x: x[1])
df_all['common_digits_in_title_num'] = df_all['common_digits_in_title_tuple'].map(lambda x: x[2])
df_all['common_digits_in_title_ratio'] = df_all['common_digits_in_title_tuple'].map(lambda x: x[3])
df_all['common_digits_in_title_jaccard'] = df_all['common_digits_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['common_digits_in_title_tuple'],axis=1)

df_all['nn_important_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['product_title_stemmed']),axis=1)
df_all['nn_important_in_title_num'] = df_all['nn_important_in_title_tuple'].map(lambda x: x[0])
df_all['nn_important_in_title_sum'] = df_all['nn_important_in_title_tuple'].map(lambda x: x[1])
df_all['nn_important_in_title_let'] = df_all['nn_important_in_title_tuple'].map(lambda x: x[2])
df_all['nn_important_in_title_numratio'] = df_all['nn_important_in_title_tuple'].map(lambda x: x[3])
df_all['nn_important_in_title_letratio'] = df_all['nn_important_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_title_tuple'],axis=1)

df_all['nn_unimportant_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_unimportant_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['product_title_stemmed']),axis=1)
df_all['nn_unimportant_in_title_num'] = df_all['nn_unimportant_in_title_tuple'].map(lambda x: x[0])
df_all['nn_unimportant_in_title_let'] = df_all['nn_unimportant_in_title_tuple'].map(lambda x: x[2])
df_all['nn_unimportant_in_title_letratio'] = df_all['nn_unimportant_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_unimportant_in_title_tuple'],axis=1)

"""
df_all['vbg_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(vbg_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['product_title_stemmed']),axis=1)
df_all['vbg_in_title_num'] = df_all['vbg_in_title_tuple'].map(lambda x: x[0])
df_all['vbg_in_title_sum'] = df_all['vbg_in_title_tuple'].map(lambda x: x[1])
df_all['vbg_in_title_let'] = df_all['vbg_in_title_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['vbg_in_title_tuple'],axis=1)

df_all['jj_rb_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(jj_rb_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['product_title_stemmed']),axis=1)
df_all['jj_rb_in_title_num'] = df_all['jj_rb_in_title_tuple'].map(lambda x: x[0])
df_all['jj_rb_in_title_sum'] = df_all['jj_rb_in_title_tuple'].map(lambda x: x[1])
df_all['jj_rb_in_title_let'] = df_all['jj_rb_in_title_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['jj_rb_in_title_tuple'],axis=1)
"""

df_all['nn_important_in_nn_important_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_important_words(x['product_title_tokens']),stoplist=stoplist_wo_can)),axis=1)
df_all['nn_important_in_nn_important_in_title_num'] = df_all['nn_important_in_nn_important_in_title_tuple'].map(lambda x: x[0])
df_all['nn_important_in_nn_important_in_title_sum'] = df_all['nn_important_in_nn_important_in_title_tuple'].map(lambda x: x[1])
df_all['nn_important_in_nn_important_in_title_let'] = df_all['nn_important_in_nn_important_in_title_tuple'].map(lambda x: x[2])
df_all['nn_important_in_nn_important_in_title_letratio'] = df_all['nn_important_in_nn_important_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_nn_important_in_title_tuple'],axis=1)

df_all['nn_important_in_nn_unimportant_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_unimportant_words(x['product_title_tokens']),stoplist=stoplist_wo_can)),axis=1)
df_all['nn_important_in_nn_unimportant_in_title_num'] = df_all['nn_important_in_nn_unimportant_in_title_tuple'].map(lambda x: x[0])
df_all['nn_important_in_nn_unimportant_in_title_sum'] = df_all['nn_important_in_nn_unimportant_in_title_tuple'].map(lambda x: x[1])
df_all['nn_important_in_nn_unimportant_in_title_let'] = df_all['nn_important_in_nn_unimportant_in_title_tuple'].map(lambda x: x[2])
df_all['nn_important_in_nn_unimportant_in_title_letratio'] = df_all['nn_important_in_nn_unimportant_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_nn_unimportant_in_title_tuple'],axis=1)

df_all['nn_unimportant_in_nn_important_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_unimportant_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_important_words(x['product_title_tokens']),stoplist=stoplist_wo_can)),axis=1)
df_all['nn_unimportant_in_nn_important_in_title_num'] = df_all['nn_unimportant_in_nn_important_in_title_tuple'].map(lambda x: x[0])
df_all['nn_unimportant_in_nn_important_in_title_sum'] = df_all['nn_unimportant_in_nn_important_in_title_tuple'].map(lambda x: x[1])
df_all['nn_unimportant_in_nn_important_in_title_let'] = df_all['nn_unimportant_in_nn_important_in_title_tuple'].map(lambda x: x[2])
df_all['nn_unimportant_in_nn_important_in_title_letratio'] = df_all['nn_unimportant_in_nn_important_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_unimportant_in_nn_important_in_title_tuple'],axis=1)

df_all['jj_rb_in_jj_rb_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(jj_rb_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(jj_rb_words(x['product_title_tokens']),stoplist=stoplist_wo_can)),axis=1)
df_all['jj_rb_in_jj_rb_in_title_num'] = df_all['jj_rb_in_jj_rb_in_title_tuple'].map(lambda x: x[0])
df_all['jj_rb_in_jj_rb_in_title_sum'] = df_all['jj_rb_in_jj_rb_in_title_tuple'].map(lambda x: x[1])
df_all['jj_rb_in_jj_rb_in_title_let'] = df_all['jj_rb_in_jj_rb_in_title_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['jj_rb_in_jj_rb_in_title_tuple'],axis=1)

df_all['vbg_in_vbg_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(vbg_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(vbg_words(x['product_title_tokens']),stoplist=stoplist_wo_can)),axis=1)
df_all['vbg_in_vbg_in_title_num'] = df_all['vbg_in_vbg_in_title_tuple'].map(lambda x: x[0])
df_all['vbg_in_vbg_in_title_sum'] = df_all['vbg_in_vbg_in_title_tuple'].map(lambda x: x[1])
df_all['vbg_in_vbg_in_title_let'] = df_all['vbg_in_vbg_in_title_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['vbg_in_vbg_in_title_tuple'],axis=1)
print 'words_in_title time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


###################################
###################################

################################
### query vs product description


df_all['wordFor_in_description_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_for_stemmed'],x['product_description_stemmed'],string_only=True),axis=1)
df_all['wordFor_in_description_string_only_num'] = df_all['wordFor_in_description_string_only_tuple'].map(lambda x: x[0])
df_all['wordFor_in_description_string_only_let'] = df_all['wordFor_in_description_string_only_tuple'].map(lambda x: x[2])
df_all['wordFor_in_description_string_only_letratio'] = df_all['wordFor_in_description_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['wordFor_in_description_string_only_tuple'],axis=1)

df_all['wordWith_in_description_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_with_stemmed'],x['product_description_stemmed'],string_only=True),axis=1)
df_all['wordWith_in_description_string_only_num'] = df_all['wordWith_in_description_string_only_tuple'].map(lambda x: x[0])
df_all['wordWith_in_description_string_only_let'] = df_all['wordWith_in_description_string_only_tuple'].map(lambda x: x[2])
df_all['wordWith_in_description_string_only_letratio'] = df_all['wordWith_in_description_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['wordWith_in_description_string_only_tuple'],axis=1)

df_all['query_in_description']=df_all.apply(lambda x: \
            query_in_text(x['search_term_stemmed'],x['product_description_stemmed']),axis=1)

df_all['word_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_stemmed'],x['product_description_stemmed']),axis=1)
df_all['word_in_description_num'] = df_all['word_in_description_tuple'].map(lambda x: x[0])
df_all['word_in_description_sum'] = df_all['word_in_description_tuple'].map(lambda x: x[1])
df_all['word_in_description_let'] = df_all['word_in_description_tuple'].map(lambda x: x[2])
df_all['word_in_description_numratio'] = df_all['word_in_description_tuple'].map(lambda x: x[3])
df_all['word_in_description_letratio'] = df_all['word_in_description_tuple'].map(lambda x: x[4])
df_all['word_in_description_string'] = df_all['word_in_description_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['word_in_description_tuple'],axis=1)

df_all['word_in_description_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_stemmed'],x['product_description_stemmed'],string_only=True),axis=1)
df_all['word_in_description_string_only_num'] = df_all['word_in_description_string_only_tuple'].map(lambda x: x[0])
df_all['word_in_description_string_only_sum'] = df_all['word_in_description_string_only_tuple'].map(lambda x: x[1])
df_all['word_in_description_string_only_let'] = df_all['word_in_description_string_only_tuple'].map(lambda x: x[2])
df_all['word_in_description_string_only_numratio'] = df_all['word_in_description_string_only_tuple'].map(lambda x: x[3])
df_all['word_in_description_string_only_letratio'] = df_all['word_in_description_string_only_tuple'].map(lambda x: x[4])
df_all['word_in_description_string_only_string'] = df_all['word_in_description_string_only_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['word_in_description_string_only_tuple'],axis=1)

df_all['word_in_description_w_dash']=df_all.apply(lambda x: \
            str_common_word(words_w_dash(x['search_term_stemmed']),words_w_dash(x['product_description_stemmed']))[0],axis=1)

df_all['two_words_in_description_tuple']=df_all.apply(lambda x: \
            str_2common_words(x['search_term_stemmed'],x['product_description_stemmed']),axis=1)
df_all['two_words_in_description_num'] = df_all['two_words_in_description_tuple'].map(lambda x: x[0])
df_all['two_words_in_description_sum'] = df_all['two_words_in_description_tuple'].map(lambda x: x[1])
df_all['two_words_in_description_let'] = df_all['two_words_in_description_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['two_words_in_description_tuple'],axis=1)

df_all['two_words_in_description_string_only_tuple']=df_all.apply(lambda x: \
            str_2common_words(x['search_term_stemmed'],x['product_description_stemmed'],string_only=True),axis=1)
df_all['two_words_in_description_string_only_num'] = df_all['two_words_in_description_string_only_tuple'].map(lambda x: x[0])
df_all['two_words_in_description_string_only_sum'] = df_all['two_words_in_description_string_only_tuple'].map(lambda x: x[1])
df_all['two_words_in_description_string_only_let'] = df_all['two_words_in_description_string_only_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['two_words_in_description_string_only_tuple'],axis=1)

df_all['common_digits_in_description_tuple']=df_all.apply(lambda x: \
            str_common_digits(x['search_term_stemmed'],x['product_description_stemmed']),axis=1)
df_all['len_of_digits_in_query'] = df_all['common_digits_in_description_tuple'].map(lambda x: x[0])
df_all['len_of_digits_in_description'] = df_all['common_digits_in_description_tuple'].map(lambda x: x[1])
df_all['common_digits_in_description_num'] = df_all['common_digits_in_description_tuple'].map(lambda x: x[2])
df_all['common_digits_in_description_ratio'] = df_all['common_digits_in_description_tuple'].map(lambda x: x[3])
df_all['common_digits_in_description_jaccard'] = df_all['common_digits_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['common_digits_in_description_tuple'],axis=1)

df_all['nn_important_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['product_description_stemmed']),axis=1)
df_all['nn_important_in_description_num'] = df_all['nn_important_in_description_tuple'].map(lambda x: x[0])
df_all['nn_important_in_description_sum'] = df_all['nn_important_in_description_tuple'].map(lambda x: x[1])
df_all['nn_important_in_description_let'] = df_all['nn_important_in_description_tuple'].map(lambda x: x[2])
df_all['nn_important_in_description_numratio'] = df_all['nn_important_in_description_tuple'].map(lambda x: x[3])
df_all['nn_important_in_description_letratio'] = df_all['nn_important_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_description_tuple'],axis=1)

df_all['nn_unimportant_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_unimportant_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['product_description_stemmed']),axis=1)
df_all['nn_unimportant_in_description_num'] = df_all['nn_unimportant_in_description_tuple'].map(lambda x: x[0])
df_all['nn_unimportant_in_description_let'] = df_all['nn_unimportant_in_description_tuple'].map(lambda x: x[2])
df_all['nn_unimportant_in_description_letratio'] = df_all['nn_unimportant_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_unimportant_in_description_tuple'],axis=1)


df_all['nn_important_in_nn_important_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_important_words(x['product_description_tokens']))),axis=1)
df_all['nn_important_in_nn_important_in_description_num'] = df_all['nn_important_in_nn_important_in_description_tuple'].map(lambda x: x[0])
df_all['nn_important_in_nn_important_in_description_sum'] = df_all['nn_important_in_nn_important_in_description_tuple'].map(lambda x: x[1])
df_all['nn_important_in_nn_important_in_description_let'] = df_all['nn_important_in_nn_important_in_description_tuple'].map(lambda x: x[2])
df_all['nn_important_in_nn_important_in_description_letratio'] = df_all['nn_important_in_nn_important_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_nn_important_in_description_tuple'],axis=1)

df_all['nn_important_in_nn_unimportant_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_unimportant_words(x['product_description_tokens']))),axis=1)
df_all['nn_important_in_nn_unimportant_in_description_num'] = df_all['nn_important_in_nn_unimportant_in_description_tuple'].map(lambda x: x[0])
df_all['nn_important_in_nn_unimportant_in_description_sum'] = df_all['nn_important_in_nn_unimportant_in_description_tuple'].map(lambda x: x[1])
df_all['nn_important_in_nn_unimportant_in_description_let'] = df_all['nn_important_in_nn_unimportant_in_description_tuple'].map(lambda x: x[2])
df_all['nn_important_in_nn_unimportant_in_description_letratio'] = df_all['nn_important_in_nn_unimportant_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_nn_unimportant_in_description_tuple'],axis=1)

df_all['nn_unimportant_in_nn_important_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_unimportant_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_important_words(x['product_description_tokens']))),axis=1)
df_all['nn_unimportant_in_nn_important_in_description_num'] = df_all['nn_unimportant_in_nn_important_in_description_tuple'].map(lambda x: x[0])
df_all['nn_unimportant_in_nn_important_in_description_sum'] = df_all['nn_unimportant_in_nn_important_in_description_tuple'].map(lambda x: x[1])
df_all['nn_unimportant_in_nn_important_in_description_let'] = df_all['nn_unimportant_in_nn_important_in_description_tuple'].map(lambda x: x[2])
df_all['nn_unimportant_in_nn_important_in_description_letratio'] = df_all['nn_unimportant_in_nn_important_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_unimportant_in_nn_important_in_description_tuple'],axis=1)

df_all['jj_rb_in_jj_rb_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(jj_rb_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(jj_rb_words(x['product_description_tokens']))),axis=1)
df_all['jj_rb_in_jj_rb_in_description_num'] = df_all['jj_rb_in_jj_rb_in_description_tuple'].map(lambda x: x[0])
df_all['jj_rb_in_jj_rb_in_description_sum'] = df_all['jj_rb_in_jj_rb_in_description_tuple'].map(lambda x: x[1])
df_all['jj_rb_in_jj_rb_in_description_let'] = df_all['jj_rb_in_jj_rb_in_description_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['jj_rb_in_jj_rb_in_description_tuple'],axis=1)

df_all['vbg_in_vbg_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(vbg_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(vbg_words(x['product_description_tokens']))),axis=1)
df_all['vbg_in_vbg_in_description_num'] = df_all['vbg_in_vbg_in_description_tuple'].map(lambda x: x[0])
df_all['vbg_in_vbg_in_description_sum'] = df_all['vbg_in_vbg_in_description_tuple'].map(lambda x: x[1])
df_all['vbg_in_vbg_in_description_let'] = df_all['vbg_in_vbg_in_description_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['vbg_in_vbg_in_description_tuple'],axis=1)
print 'words_in_description time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()




###################################
###################################


################################
### query vs attribute bullets

df_all['wordFor_in_bullets_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_for_stemmed'],x['attribute_bullets_stemmed'],string_only=True),axis=1)
df_all['wordFor_in_bullets_string_only_num'] = df_all['wordFor_in_bullets_string_only_tuple'].map(lambda x: x[0])
df_all['wordFor_in_bullets_string_only_let'] = df_all['wordFor_in_bullets_string_only_tuple'].map(lambda x: x[2])
df_all['wordFor_in_bullets_string_only_letratio'] = df_all['wordFor_in_bullets_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['wordFor_in_bullets_string_only_tuple'],axis=1)

df_all['wordWith_in_bullets_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_with_stemmed'],x['attribute_bullets_stemmed'],string_only=True),axis=1)
df_all['wordWith_in_bullets_string_only_num'] = df_all['wordWith_in_bullets_string_only_tuple'].map(lambda x: x[0])
df_all['wordWith_in_bullets_string_only_let'] = df_all['wordWith_in_bullets_string_only_tuple'].map(lambda x: x[2])
df_all['wordWith_in_bullets_string_only_letratio'] = df_all['wordWith_in_bullets_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['wordWith_in_bullets_string_only_tuple'],axis=1)

df_all['query_in_bullets']=df_all.apply(lambda x: \
            query_in_text(x['search_term_stemmed'],x['attribute_bullets_stemmed']),axis=1)

df_all['word_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_stemmed'],x['attribute_bullets_stemmed']),axis=1)
df_all['word_in_bullets_num'] = df_all['word_in_bullets_tuple'].map(lambda x: x[0])
df_all['word_in_bullets_sum'] = df_all['word_in_bullets_tuple'].map(lambda x: x[1])
df_all['word_in_bullets_let'] = df_all['word_in_bullets_tuple'].map(lambda x: x[2])
df_all['word_in_bullets_numratio'] = df_all['word_in_bullets_tuple'].map(lambda x: x[3])
df_all['word_in_bullets_letratio'] = df_all['word_in_bullets_tuple'].map(lambda x: x[4])
df_all['word_in_bullets_string'] = df_all['word_in_bullets_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['word_in_bullets_tuple'],axis=1)

df_all['word_in_bullets_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_stemmed'],x['attribute_bullets_stemmed'],string_only=True),axis=1)
df_all['word_in_bullets_string_only_num'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[0])
df_all['word_in_bullets_string_only_sum'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[1])
df_all['word_in_bullets_string_only_let'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[2])
df_all['word_in_bullets_string_only_numratio'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[3])
df_all['word_in_bullets_string_only_letratio'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[4])
df_all['word_in_bullets_string_only_string'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['word_in_bullets_string_only_tuple'],axis=1)

df_all['word_in_bullets_w_dash']=df_all.apply(lambda x: \
            str_common_word(words_w_dash(x['search_term_stemmed']),words_w_dash(x['attribute_bullets_stemmed']))[0],axis=1)

df_all['two_words_in_bullets_tuple']=df_all.apply(lambda x: \
            str_2common_words(x['search_term_stemmed'],x['attribute_bullets_stemmed']),axis=1)
df_all['two_words_in_bullets_num'] = df_all['two_words_in_bullets_tuple'].map(lambda x: x[0])
df_all['two_words_in_bullets_sum'] = df_all['two_words_in_bullets_tuple'].map(lambda x: x[1])
df_all['two_words_in_bullets_let'] = df_all['two_words_in_bullets_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['two_words_in_bullets_tuple'],axis=1)

df_all['two_words_in_bullets_string_only_tuple']=df_all.apply(lambda x: \
            str_2common_words(x['search_term_stemmed'],x['attribute_bullets_stemmed'],string_only=True),axis=1)
df_all['two_words_in_bullets_string_only_num'] = df_all['two_words_in_bullets_string_only_tuple'].map(lambda x: x[0])
df_all['two_words_in_bullets_string_only_sum'] = df_all['two_words_in_bullets_string_only_tuple'].map(lambda x: x[1])
df_all['two_words_in_bullets_string_only_let'] = df_all['two_words_in_bullets_string_only_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['two_words_in_bullets_string_only_tuple'],axis=1)

df_all['common_digits_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_digits(x['search_term_stemmed'],x['attribute_bullets_stemmed']),axis=1)
df_all['len_of_digits_in_query'] = df_all['common_digits_in_bullets_tuple'].map(lambda x: x[0])
df_all['len_of_digits_in_bullets'] = df_all['common_digits_in_bullets_tuple'].map(lambda x: x[1])
df_all['common_digits_in_bullets_num'] = df_all['common_digits_in_bullets_tuple'].map(lambda x: x[2])
df_all['common_digits_in_bullets_ratio'] = df_all['common_digits_in_bullets_tuple'].map(lambda x: x[3])
df_all['common_digits_in_bullets_jaccard'] = df_all['common_digits_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['common_digits_in_bullets_tuple'],axis=1)

df_all['nn_important_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['attribute_bullets_stemmed']),axis=1)
df_all['nn_important_in_bullets_num'] = df_all['nn_important_in_bullets_tuple'].map(lambda x: x[0])
df_all['nn_important_in_bullets_sum'] = df_all['nn_important_in_bullets_tuple'].map(lambda x: x[1])
df_all['nn_important_in_bullets_let'] = df_all['nn_important_in_bullets_tuple'].map(lambda x: x[2])
df_all['nn_important_in_bullets_numratio'] = df_all['nn_important_in_bullets_tuple'].map(lambda x: x[3])
df_all['nn_important_in_bullets_letratio'] = df_all['nn_important_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_bullets_tuple'],axis=1)

df_all['nn_unimportant_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_unimportant_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['attribute_bullets_stemmed']),axis=1)
df_all['nn_unimportant_in_bullets_num'] = df_all['nn_unimportant_in_bullets_tuple'].map(lambda x: x[0])
df_all['nn_unimportant_in_bullets_let'] = df_all['nn_unimportant_in_bullets_tuple'].map(lambda x: x[2])
df_all['nn_unimportant_in_bullets_letratio'] = df_all['nn_unimportant_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_unimportant_in_bullets_tuple'],axis=1)


df_all['nn_important_in_nn_important_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_important_words(x['attribute_bullets_tokens']))),axis=1)
df_all['nn_important_in_nn_important_in_bullets_num'] = df_all['nn_important_in_nn_important_in_bullets_tuple'].map(lambda x: x[0])
df_all['nn_important_in_nn_important_in_bullets_sum'] = df_all['nn_important_in_nn_important_in_bullets_tuple'].map(lambda x: x[1])
df_all['nn_important_in_nn_important_in_bullets_let'] = df_all['nn_important_in_nn_important_in_bullets_tuple'].map(lambda x: x[2])
df_all['nn_important_in_nn_important_in_bullets_letratio'] = df_all['nn_important_in_nn_important_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_nn_important_in_bullets_tuple'],axis=1)

df_all['nn_important_in_nn_unimportant_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_unimportant_words(x['attribute_bullets_tokens']))),axis=1)
df_all['nn_important_in_nn_unimportant_in_bullets_num'] = df_all['nn_important_in_nn_unimportant_in_bullets_tuple'].map(lambda x: x[0])
df_all['nn_important_in_nn_unimportant_in_bullets_sum'] = df_all['nn_important_in_nn_unimportant_in_bullets_tuple'].map(lambda x: x[1])
df_all['nn_important_in_nn_unimportant_in_bullets_let'] = df_all['nn_important_in_nn_unimportant_in_bullets_tuple'].map(lambda x: x[2])
df_all['nn_important_in_nn_unimportant_in_bullets_letratio'] = df_all['nn_important_in_nn_unimportant_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_nn_unimportant_in_bullets_tuple'],axis=1)

df_all['nn_unimportant_in_nn_important_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_unimportant_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_important_words(x['attribute_bullets_tokens']))),axis=1)
df_all['nn_unimportant_in_nn_important_in_bullets_num'] = df_all['nn_unimportant_in_nn_important_in_bullets_tuple'].map(lambda x: x[0])
df_all['nn_unimportant_in_nn_important_in_bullets_sum'] = df_all['nn_unimportant_in_nn_important_in_bullets_tuple'].map(lambda x: x[1])
df_all['nn_unimportant_in_nn_important_in_bullets_let'] = df_all['nn_unimportant_in_nn_important_in_bullets_tuple'].map(lambda x: x[2])
df_all['nn_unimportant_in_nn_important_in_bullets_letratio'] = df_all['nn_unimportant_in_nn_important_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_unimportant_in_nn_important_in_bullets_tuple'],axis=1)

df_all['jj_rb_in_jj_rb_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(jj_rb_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(jj_rb_words(x['attribute_bullets_tokens']))),axis=1)
df_all['jj_rb_in_jj_rb_in_bullets_num'] = df_all['jj_rb_in_jj_rb_in_bullets_tuple'].map(lambda x: x[0])
df_all['jj_rb_in_jj_rb_in_bullets_sum'] = df_all['jj_rb_in_jj_rb_in_bullets_tuple'].map(lambda x: x[1])
df_all['jj_rb_in_jj_rb_in_bullets_let'] = df_all['jj_rb_in_jj_rb_in_bullets_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['jj_rb_in_jj_rb_in_bullets_tuple'],axis=1)

df_all['vbg_in_vbg_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(vbg_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(vbg_words(x['attribute_bullets_tokens']))),axis=1)
df_all['vbg_in_vbg_in_bullets_num'] = df_all['vbg_in_vbg_in_bullets_tuple'].map(lambda x: x[0])
df_all['vbg_in_vbg_in_bullets_sum'] = df_all['vbg_in_vbg_in_bullets_tuple'].map(lambda x: x[1])
df_all['vbg_in_vbg_in_bullets_let'] = df_all['vbg_in_vbg_in_bullets_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['vbg_in_vbg_in_bullets_tuple'],axis=1)
print 'words_in_bullets time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

















#################################################################
### STEP 5: Process the most important words (keywords)
#################################################################



############################################################
### count common words/letters and jaccard coefficients


t0 = time()

df_all['keyword_in_titlekeys_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_keys_stemmed'],x['product_title_keys_stemmed']),axis=1)
df_all['keyword_in_titlekeys_num'] = df_all['keyword_in_titlekeys_tuple'].map(lambda x: x[0])
df_all['keyword_in_titlekeys_sum'] = df_all['keyword_in_titlekeys_tuple'].map(lambda x: x[1])
df_all['keyword_in_titlekeys_let'] = df_all['keyword_in_titlekeys_tuple'].map(lambda x: x[2])
df_all['keyword_in_titlekeys_numratio'] = df_all['keyword_in_titlekeys_tuple'].map(lambda x: x[3])
df_all['keyword_in_titlekeys_letratio'] = df_all['keyword_in_titlekeys_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['keyword_in_titlekeys_tuple'],axis=1)

df_all['keyword_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_keys_stemmed'],x['product_description_stemmed']),axis=1)
df_all['keyword_in_description_num'] = df_all['keyword_in_description_tuple'].map(lambda x: x[0])
df_all['keyword_in_description_sum'] = df_all['keyword_in_description_tuple'].map(lambda x: x[1])
df_all['keyword_in_description_let'] = df_all['keyword_in_description_tuple'].map(lambda x: x[2])
df_all['keyword_in_description_numratio'] = df_all['keyword_in_description_tuple'].map(lambda x: x[3])
df_all['keyword_in_description_letratio'] = df_all['keyword_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['keyword_in_description_tuple'],axis=1)

df_all['keyword_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_keys_stemmed'],x['attribute_bullets_stemmed']),axis=1)
df_all['keyword_in_bullets_num'] = df_all['keyword_in_bullets_tuple'].map(lambda x: x[0])
df_all['keyword_in_bullets_sum'] = df_all['keyword_in_bullets_tuple'].map(lambda x: x[1])
df_all['keyword_in_bullets_let'] = df_all['keyword_in_bullets_tuple'].map(lambda x: x[2])
df_all['keyword_in_bullets_numratio'] = df_all['keyword_in_bullets_tuple'].map(lambda x: x[3])
df_all['keyword_in_bullets_letratio'] = df_all['keyword_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['keyword_in_bullets_tuple'],axis=1)


df_all['jaccard_keyword_in_titlekeys_tuple']=df_all.apply(lambda x: \
            str_jaccard(x['search_term_keys_stemmed'],x['product_title_keys_stemmed']),axis=1)
df_all['keyword_in_titlekeys_jacnum'] = df_all['jaccard_keyword_in_titlekeys_tuple'].map(lambda x: x[0])
df_all['keyword_in_titlekeys_jaclet'] = df_all['jaccard_keyword_in_titlekeys_tuple'].map(lambda x: x[1])
df_all=df_all.drop(['jaccard_keyword_in_titlekeys_tuple'],axis=1)




def concatenate_keys(beforethekey,thekey):
    l=[beforethekey,thekey]
    if "" in l:
        l.remove("")
    if "" in l:
        l.remove("")
    return " ".join(l)

df_all['thekeys_in_title']=df_all.apply(lambda x: \
            query_in_text(concatenate_keys(x['search_term_beforethekey_stemmed'],x['search_term_thekey_stemmed']),\
                          x['product_title_stemmed']), axis=1)  
df_all['thekeys_in_description']=df_all.apply(lambda x: \
            query_in_text(concatenate_keys(x['search_term_beforethekey_stemmed'],x['search_term_thekey_stemmed']),\
                          x['product_description_stemmed']), axis=1)  
df_all['thekeys_in_bullets']=df_all.apply(lambda x: \
            query_in_text(concatenate_keys(x['search_term_beforethekey_stemmed'],x['search_term_thekey_stemmed']),\
                          x['attribute_bullets_stemmed']), axis=1)                            

df_all['thekey_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_thekey_stemmed'],x['product_description_stemmed']), axis=1) 
df_all['thekey_in_description_sum'] = df_all['thekey_in_description_tuple'].map(lambda x: x[1])
df_all['thekey_in_description_let'] = df_all['thekey_in_description_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['thekey_in_description_tuple'],axis=1)

df_all['thekey_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_thekey_stemmed'],x['attribute_bullets_stemmed']), axis=1) 
df_all['thekey_in_bullets_sum'] = df_all['thekey_in_bullets_tuple'].map(lambda x: x[1])
df_all['thekey_in_bullets_let'] = df_all['thekey_in_bullets_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['thekey_in_bullets_tuple'],axis=1)

df_all['beforethekey_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_beforethekey_stemmed'],x['product_description_stemmed']), axis=1) 
df_all['beforethekey_in_description_sum'] = df_all['beforethekey_in_description_tuple'].map(lambda x: x[1])
df_all['beforethekey_in_description_let'] = df_all['beforethekey_in_description_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['beforethekey_in_description_tuple'],axis=1)

df_all['beforethekey_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_beforethekey_stemmed'],x['attribute_bullets_stemmed']), axis=1) 
df_all['beforethekey_in_bullets_sum'] = df_all['beforethekey_in_bullets_tuple'].map(lambda x: x[1])
df_all['beforethekey_in_bullets_let'] = df_all['beforethekey_in_bullets_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['beforethekey_in_bullets_tuple'],axis=1)


df_all['thekey_in_nn_important_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_thekey_stemmed'],
                            str_stemmer_wo_parser(nn_important_words(x['product_description_tokens']))),axis=1)
df_all['thekey_in_nn_important_in_description_sum'] = df_all['thekey_in_nn_important_in_description_tuple'].map(lambda x: x[1])
df_all['thekey_in_nn_important_in_description_let'] = df_all['thekey_in_nn_important_in_description_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['thekey_in_nn_important_in_description_tuple'],axis=1)

df_all['thekey_in_nn_important_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_thekey_stemmed'],
                            str_stemmer_wo_parser(nn_important_words(x['attribute_bullets_tokens']))),axis=1)
df_all['thekey_in_nn_important_in_bullets_sum'] = df_all['thekey_in_nn_important_in_bullets_tuple'].map(lambda x: x[1])
df_all['thekey_in_nn_important_in_bullets_let'] = df_all['thekey_in_nn_important_in_bullets_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['thekey_in_nn_important_in_bullets_tuple'],axis=1)





### find whether important words match
def match_thekeywords(word1,word2):
    return int(len(word1)>0 and len(word2)>0 and (word1 in word2 or word2 in word1))

df_all['thekey_in_thekey']=df_all.apply(lambda x: \
            match_thekeywords(x['search_term_thekey_stemmed'],x['product_title_thekey_stemmed']),axis=1)
df_all['beforethekey_in_beforethekey']=df_all.apply(lambda x: \
            match_thekeywords(x['search_term_beforethekey_stemmed'],x['product_title_beforethekey_stemmed']),axis=1) 
df_all['beforethekeys_in_beforethekeys']=df_all.apply(lambda x: \
            max(match_thekeywords(x['search_term_beforethekey_stemmed'],x['product_title_beforethekey_stemmed']),\
                match_thekeywords(x['search_term_beforethekey_stemmed'],x['product_title_before2thekey_stemmed']),\
                match_thekeywords(x['search_term_before2thekey_stemmed'],x['product_title_beforethekey_stemmed']),\
                match_thekeywords(x['search_term_before2thekey_stemmed'],x['product_title_before2thekey_stemmed'])),axis=1)
df_all['thekey_in_beforethekeys']=df_all.apply(lambda x: \
            max(match_thekeywords(x['search_term_thekey_stemmed'],x['product_title_beforethekey_stemmed']),\
                match_thekeywords(x['search_term_thekey_stemmed'],x['product_title_before2thekey_stemmed'])),axis=1)
df_all['beforethekeys_in_thekey']=df_all.apply(lambda x: \
            max(match_thekeywords(x['search_term_thekey_stemmed'],x['product_title_beforethekey_stemmed']),\
                match_thekeywords(x['search_term_thekey_stemmed'],x['product_title_before2thekey_stemmed'])),axis=1)
            


#################################################################
### Use NLTK WordNet to calculate similarities between keywords
t2 = time()

##############
df_all['key_for_dict']=df_all.apply(lambda x: x['search_term_thekey']+"\t"+x['product_title_thekey'],axis=1) 
aa=list(set(list(df_all['key_for_dict'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=find_similarity(aa[i].split("\t")[0],aa[i].split("\t")[1])
    if (i % 5000)==0:
        print ""+str(i)+" out of "+str(len(aa))+" unique combinations; "+str(round((time()-t2)/60,1))+" minutes"
    
df_all['thekeys_similarity_tuple']=df_all['key_for_dict'].map(lambda x: my_dict[x] )
df_all['thekeys_pathsimilarity_max'] = df_all['thekeys_similarity_tuple'].map(lambda x: x[0])
df_all['thekeys_pathsimilarity_mean'] = df_all['thekeys_similarity_tuple'].map(lambda x: x[1])
df_all['thekeys_lchsimilarity_max'] = df_all['thekeys_similarity_tuple'].map(lambda x: x[2])
df_all['thekeys_lchsimilarity_mean'] = df_all['thekeys_similarity_tuple'].map(lambda x: x[3])
df_all['thekeys_ressimilarity_max'] = df_all['thekeys_similarity_tuple'].map(lambda x: x[4])
df_all['thekeys_ressimilarity_mean'] = df_all['thekeys_similarity_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['thekeys_similarity_tuple'],axis=1)
print 'thekeys similarity time:',round((time()-t2)/60,1) ,'minutes\n'
t2 = time()

##############
df_all['key_for_dict']=df_all.apply(lambda x: x['search_term_beforethekey']+"\t"+x['product_title_beforethekey'],axis=1) 
aa=list(set(list(df_all['key_for_dict'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=find_similarity(aa[i].split("\t")[0],aa[i].split("\t")[1],nouns=False)
    if (i % 5000)==0:
        print ""+str(i)+" out of "+str(len(aa))+" unique combinations; "+str(round((time()-t2)/60,1))+" minutes"
    
df_all['beforethekeys_similarity_tuple']=df_all['key_for_dict'].map(lambda x: my_dict[x] )
df_all['beforethekeys_pathsimilarity_max'] = df_all['beforethekeys_similarity_tuple'].map(lambda x: x[0])
df_all['beforethekeys_pathsimilarity_mean'] = df_all['beforethekeys_similarity_tuple'].map(lambda x: x[1])
df_all['beforethekeys_lchsimilarity_max'] = df_all['beforethekeys_similarity_tuple'].map(lambda x: x[2])
df_all['beforethekeys_lchsimilarity_mean'] = df_all['beforethekeys_similarity_tuple'].map(lambda x: x[3])
df_all['beforethekeys_ressimilarity_max'] = df_all['beforethekeys_similarity_tuple'].map(lambda x: x[4])
df_all['beforethekeys_ressimilarity_mean'] = df_all['beforethekeys_similarity_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['beforethekeys_similarity_tuple'],axis=1)
print 'beforethekeys similarity time:',round((time()-t2)/60,1) ,'minutes\n'
t2 = time()



##############
df_all['key_for_dict']=df_all.apply(lambda x: x['search_term_thekey']+"\t"+x['product_title_beforethekey'],axis=1) 
aa=list(set(list(df_all['key_for_dict'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=find_similarity(aa[i].split("\t")[0],aa[i].split("\t")[1],nouns=False)
    if (i % 5000)==0:
        print ""+str(i)+" out of "+str(len(aa))+" unique combinations; "+str(round((time()-t2)/60,1))+" minutes"
df_all['thekey_beforethekey_similarity_tuple']=df_all['key_for_dict'].map(lambda x: my_dict[x] )
df_all['thekey_beforethekey_pathsimilarity_max'] = df_all['thekey_beforethekey_similarity_tuple'].map(lambda x: x[0])
df_all['thekey_beforethekey_pathsimilarity_mean'] = df_all['thekey_beforethekey_similarity_tuple'].map(lambda x: x[1])
df_all['thekey_beforethekey_lchsimilarity_max'] = df_all['thekey_beforethekey_similarity_tuple'].map(lambda x: x[2])
df_all['thekey_beforethekey_lchsimilarity_mean'] = df_all['thekey_beforethekey_similarity_tuple'].map(lambda x: x[3])
df_all['thekey_beforethekey_ressimilarity_max'] = df_all['thekey_beforethekey_similarity_tuple'].map(lambda x: x[4])
df_all['thekey_beforethekey_ressimilarity_mean'] = df_all['thekey_beforethekey_similarity_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['thekey_beforethekey_similarity_tuple'],axis=1)
print 'thekey_beforethekey similarity time:',round((time()-t2)/60,1) ,'minutes\n'
t2 = time()


df_all['key_for_dict']=df_all.apply(lambda x: x['search_term_thekey']+"\t"+x['product_title_before2thekey'],axis=1) 
aa=list(set(list(df_all['key_for_dict'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=find_similarity(aa[i].split("\t")[0],aa[i].split("\t")[1],nouns=False)
    if (i % 5000)==0:
        print ""+str(i)+" out of "+str(len(aa))+" unique combinations; "+str(round((time()-t2)/60,1))+" minutes"
df_all['thekey_before2thekey_similarity_tuple']=df_all['key_for_dict'].map(lambda x: my_dict[x] )
df_all['thekey_before2thekey_pathsimilarity_max'] = df_all['thekey_before2thekey_similarity_tuple'].map(lambda x: x[0])
df_all['thekey_before2thekey_pathsimilarity_mean'] = df_all['thekey_before2thekey_similarity_tuple'].map(lambda x: x[1])
df_all['thekey_before2thekey_lchsimilarity_max'] = df_all['thekey_before2thekey_similarity_tuple'].map(lambda x: x[2])
df_all['thekey_before2thekey_lchsimilarity_mean'] = df_all['thekey_before2thekey_similarity_tuple'].map(lambda x: x[3])
df_all['thekey_before2thekey_ressimilarity_max'] = df_all['thekey_before2thekey_similarity_tuple'].map(lambda x: x[4])
df_all['thekey_before2thekey_ressimilarity_mean'] = df_all['thekey_before2thekey_similarity_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['thekey_before2thekey_similarity_tuple'],axis=1)
print 'thekey_before2thekey similarity time:',round((time()-t2)/60,1) ,'minutes\n'
t2 = time()

for var_name in ['pathsimilarity_max','pathsimilarity_mean','lchsimilarity_max','lchsimilarity_mean',
                 'ressimilarity_max', 'ressimilarity_mean']:
    df_all['thekey_beforethekeys_'+var_name] = df_all.apply(lambda x: \
        max(x['thekey_beforethekey_'+var_name], x['thekey_before2thekey_'+var_name]),axis=1)


##############
df_all['key_for_dict']=df_all.apply(lambda x: x['search_term_beforethekey']+"\t"+x['product_title_thekey'],axis=1) 
aa=list(set(list(df_all['key_for_dict'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=find_similarity(aa[i].split("\t")[0],aa[i].split("\t")[1],nouns=False)
    if (i % 5000)==0:
        print ""+str(i)+" out of "+str(len(aa))+" unique combinations; "+str(round((time()-t2)/60,1))+" minutes"
    
df_all['beforethekey_thekey_similarity_tuple']=df_all['key_for_dict'].map(lambda x: my_dict[x] )
df_all['beforethekey_thekey_pathsimilarity_max'] = df_all['beforethekey_thekey_similarity_tuple'].map(lambda x: x[0])
df_all['beforethekey_thekey_pathsimilarity_mean'] = df_all['beforethekey_thekey_similarity_tuple'].map(lambda x: x[1])
df_all['beforethekey_thekey_lchsimilarity_max'] = df_all['beforethekey_thekey_similarity_tuple'].map(lambda x: x[2])
df_all['beforethekey_thekey_lchsimilarity_mean'] = df_all['beforethekey_thekey_similarity_tuple'].map(lambda x: x[3])
df_all['beforethekey_thekey_ressimilarity_max'] = df_all['beforethekey_thekey_similarity_tuple'].map(lambda x: x[4])
df_all['beforethekey_thekey_ressimilarity_mean'] = df_all['beforethekey_thekey_similarity_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['beforethekey_thekey_similarity_tuple'],axis=1)
print 'beforethekey_thekey similarity time:',round((time()-t2)/60,1) ,'minutes\n'
t2 = time()


df_all['key_for_dict']=df_all.apply(lambda x: x['search_term_before2thekey']+"\t"+x['product_title_thekey'],axis=1) 
aa=list(set(list(df_all['key_for_dict'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=find_similarity(aa[i].split("\t")[0],aa[i].split("\t")[1],nouns=False)
    if (i % 5000)==0:
        print ""+str(i)+" out of "+str(len(aa))+" unique combinations; "+str(round((time()-t2)/60,1))+" minutes"
    
df_all['before2thekey_thekey_similarity_tuple']=df_all['key_for_dict'].map(lambda x: my_dict[x] )
df_all['before2thekey_thekey_pathsimilarity_max'] = df_all['before2thekey_thekey_similarity_tuple'].map(lambda x: x[0])
df_all['before2thekey_thekey_pathsimilarity_mean'] = df_all['before2thekey_thekey_similarity_tuple'].map(lambda x: x[1])
df_all['before2thekey_thekey_lchsimilarity_max'] = df_all['before2thekey_thekey_similarity_tuple'].map(lambda x: x[2])
df_all['before2thekey_thekey_lchsimilarity_mean'] = df_all['before2thekey_thekey_similarity_tuple'].map(lambda x: x[3])
df_all['before2thekey_thekey_ressimilarity_max'] = df_all['before2thekey_thekey_similarity_tuple'].map(lambda x: x[4])
df_all['before2thekey_thekey_ressimilarity_mean'] = df_all['before2thekey_thekey_similarity_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['before2thekey_thekey_similarity_tuple'],axis=1)
print 'before2thekey_thekey similarity time:',round((time()-t2)/60,1) ,'minutes\n'
t2 = time()

for var_name in ['pathsimilarity_max','pathsimilarity_mean','lchsimilarity_max','lchsimilarity_mean',
                 'ressimilarity_max', 'ressimilarity_mean']:
    df_all['beforethekeys_thekey_'+var_name] = df_all.apply(lambda x: \
        max(x['beforethekey_thekey_'+var_name], x['before2thekey_thekey_'+var_name]),axis=1)


##############
df_all['key_for_dict']=df_all.apply(lambda x: x['search_term_thekey']+"\t"+x['search_term_beforethekey'],axis=1) 
aa=list(set(list(df_all['key_for_dict'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=find_similarity(aa[i].split("\t")[0],aa[i].split("\t")[1],nouns=False)
    if (i % 5000)==0:
        print ""+str(i)+" out of "+str(len(aa))+" unique combinations; "+str(round((time()-t2)/60,1))+" minutes"
    
df_all['query_similarity_tuple']=df_all['key_for_dict'].map(lambda x: my_dict[x] )
df_all['query_pathsimilarity_max'] = df_all['query_similarity_tuple'].map(lambda x: x[0])
df_all['query_pathsimilarity_mean'] = df_all['query_similarity_tuple'].map(lambda x: x[1])
df_all['query_lchhsimilarity_max'] = df_all['query_similarity_tuple'].map(lambda x: x[2])
df_all['query_lchsimilarity_mean'] = df_all['query_similarity_tuple'].map(lambda x: x[3])
df_all['query_ressimilarity_max'] = df_all['query_similarity_tuple'].map(lambda x: x[4])
df_all['query_ressimilarity_mean'] = df_all['query_similarity_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['query_similarity_tuple'],axis=1)
print 'query similarity time:',round((time()-t2)/60,1) ,'minutes\n'
t2 = time()


##############
df_all['key_for_dict']=df_all.apply(lambda x: x['product_title_thekey']+"\t"+x['product_title_beforethekey'],axis=1) 
aa=list(set(list(df_all['key_for_dict'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=find_similarity(aa[i].split("\t")[0],aa[i].split("\t")[1],nouns=False)
    if (i % 5000)==0:
        print ""+str(i)+" out of "+str(len(aa))+" unique combinations; "+str(round((time()-t2)/60,1))+" minutes"
    
df_all['title_similarity_tuple']=df_all['key_for_dict'].map(lambda x: my_dict[x] )
df_all['title_pathsimilarity_max'] = df_all['title_similarity_tuple'].map(lambda x: x[0])
df_all['title_pathsimilarity_mean'] = df_all['title_similarity_tuple'].map(lambda x: x[1])
df_all['title_lchsimilarity_max'] = df_all['title_similarity_tuple'].map(lambda x: x[2])
df_all['title_lchsimilarity_mean'] = df_all['title_similarity_tuple'].map(lambda x: x[3])
df_all['title_ressimilarity_max'] = df_all['title_similarity_tuple'].map(lambda x: x[4])
df_all['title_ressimilarity_mean'] = df_all['title_similarity_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['title_similarity_tuple'],axis=1)
print 'title similarity time:',round((time()-t2)/60,1) ,'minutes\n'
t2 = time()

##############
df_all['key_for_dict']=df_all.apply(lambda x: x['search_term_before2thekey']+"\t"+x['product_title_beforethekey'],axis=1) 
aa=list(set(list(df_all['key_for_dict'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=find_similarity(aa[i].split("\t")[0],aa[i].split("\t")[1],nouns=False)
    if (i % 5000)==0:
        print ""+str(i)+" out of "+str(len(aa))+" unique combinations; "+str(round((time()-t2)/60,1))+" minutes"
    
df_all['before2thekey_beforethekey_similarity_tuple']=df_all['key_for_dict'].map(lambda x: my_dict[x] )
df_all['before2thekey_beforethekey_pathsimilarity_max'] = df_all['before2thekey_beforethekey_similarity_tuple'].map(lambda x: x[0])
df_all['before2thekey_beforethekey_pathsimilarity_mean'] = df_all['before2thekey_beforethekey_similarity_tuple'].map(lambda x: x[1])
df_all['before2thekey_beforethekey_lchsimilarity_max'] = df_all['before2thekey_beforethekey_similarity_tuple'].map(lambda x: x[2])
df_all['before2thekey_beforethekey_lchsimilarity_mean'] = df_all['before2thekey_beforethekey_similarity_tuple'].map(lambda x: x[3])
df_all['before2thekey_beforethekey_ressimilarity_max'] = df_all['before2thekey_beforethekey_similarity_tuple'].map(lambda x: x[4])
df_all['before2thekey_beforethekey_ressimilarity_mean'] = df_all['before2thekey_beforethekey_similarity_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['before2thekey_beforethekey_similarity_tuple'],axis=1)
print 'before2thekey_beforethekey similarity time:',round((time()-t2)/60,1) ,'minutes\n'
t2 = time()


##############
df_all['key_for_dict']=df_all.apply(lambda x: x['search_term_beforethekey']+"\t"+x['product_title_before2thekey'],axis=1) 
aa=list(set(list(df_all['key_for_dict'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=find_similarity(aa[i].split("\t")[0],aa[i].split("\t")[1],nouns=False)
    if (i % 5000)==0:
        print ""+str(i)+" out of "+str(len(aa))+" unique combinations; "+str(round((time()-t2)/60,1))+" minutes"
    
df_all['beforethekey_before2thekey_similarity_tuple']=df_all['key_for_dict'].map(lambda x: my_dict[x] )
df_all['beforethekey_before2thekey_pathsimilarity_max'] = df_all['beforethekey_before2thekey_similarity_tuple'].map(lambda x: x[0])
df_all['beforethekey_before2thekey_pathsimilarity_mean'] = df_all['beforethekey_before2thekey_similarity_tuple'].map(lambda x: x[1])
df_all['beforethekey_before2thekey_lchsimilarity_max'] = df_all['beforethekey_before2thekey_similarity_tuple'].map(lambda x: x[2])
df_all['beforethekey_before2thekey_lchsimilarity_mean'] = df_all['beforethekey_before2thekey_similarity_tuple'].map(lambda x: x[3])
df_all['beforethekey_before2thekey_ressimilarity_max'] = df_all['beforethekey_before2thekey_similarity_tuple'].map(lambda x: x[4])
df_all['beforethekey_before2thekey_ressimilarity_mean'] = df_all['beforethekey_before2thekey_similarity_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['beforethekey_before2thekey_similarity_tuple'],axis=1)
print 'beforethekey_before2thekey similarity time:',round((time()-t2)/60,1) ,'minutes\n'
t2 = time()

df_all=df_all.drop(['key_for_dict'],axis=1)

print 'process key words time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()





#################################################################
### STEP 6: Estimate how similar strings in query and text are
#################################################################

### We use didifflib.SequenceMatcher() to measure similarities
### at the char level (vs. word level).
### See seq_matcher function for more details.

df_all['seqmatch_title_tuple']=df_all.apply(lambda x: \
            seq_matcher(x['search_term_stemmed'],x['product_title_stemmed']),axis=1)
df_all['seqmatch_title_ratio'] = df_all['seqmatch_title_tuple'].map(lambda x: x[0])
df_all['seqmatch_title_ratioscaled'] = df_all['seqmatch_title_tuple'].map(lambda x: x[1])
df_all=df_all.drop(['seqmatch_title_tuple'],axis=1)


df_all['seqmatch_description_tuple']=df_all.apply(lambda x: \
            seq_matcher(x['search_term_stemmed'],x['product_description_stemmed']),axis=1)
df_all['seqmatch_description_ratio'] = df_all['seqmatch_description_tuple'].map(lambda x: x[0])
df_all['seqmatch_description_ratioscaled'] = df_all['seqmatch_description_tuple'].map(lambda x: x[1])
df_all=df_all.drop(['seqmatch_description_tuple'],axis=1)

df_all['seqmatch_bullets_tuple']=df_all.apply(lambda x: \
            seq_matcher(x['search_term_stemmed'],x['attribute_bullets_stemmed']),axis=1)
df_all['seqmatch_bullets_ratio'] = df_all['seqmatch_bullets_tuple'].map(lambda x: x[0])
df_all['seqmatch_bullets_ratioscaled'] = df_all['seqmatch_bullets_tuple'].map(lambda x: x[1])
df_all=df_all.drop(['seqmatch_bullets_tuple'],axis=1)

df_all['seqmatch_desc&bullets_tuple']=df_all.apply(lambda x: \
            seq_matcher(x['search_term_stemmed'],x['product_description_stemmed']+" "+x['attribute_bullets_stemmed']),axis=1)
df_all['seqmatch_desc&bullets_ratio'] = df_all['seqmatch_desc&bullets_tuple'].map(lambda x: x[0])
df_all['seqmatch_desc&bullets_ratioscaled'] = df_all['seqmatch_desc&bullets_tuple'].map(lambda x: x[1])
df_all=df_all.drop(['seqmatch_desc&bullets_tuple'],axis=1)

print 'sequence match time:',round((time()-t0)/60,1) ,'minutes\n'



#################################################################
### STEP 7: Some TFIDF features
#################################################################

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from scipy.sparse import csr_matrix

vectorizer_title =  TfidfVectorizer(stop_words='english',max_df=0.5) 
vectorizer_description =  TfidfVectorizer(stop_words='english',max_df=0.5) 
vectorizer_bullets =  TfidfVectorizer(stop_words='english',max_df=0.5) 

features_title = vectorizer_title.fit_transform(list(set(list(df_all['product_title_stemmed'])))) 
features_description = vectorizer_description.fit_transform(list(set(list(df_all['product_description_stemmed'])))) 
features_bullets = vectorizer_bullets.fit_transform(list(set(list(df_all['attribute_bullets_stemmed'])))) 

tfidf_title = vectorizer_title.transform(df_all['search_term_stemmed']) 
tfidf_description = vectorizer_description.transform(df_all['search_term_stemmed']) 
tfidf_bullets = vectorizer_bullets.transform(df_all['search_term_stemmed']) 

tfidf_title_querythekey = vectorizer_title.transform(df_all['search_term_thekey_stemmed'])
tfidf_title_querybeforethekey = vectorizer_title.transform(df_all['search_term_beforethekey_stemmed'])


tfidf_matchtitle = vectorizer_title.transform(df_all['word_in_title_string'])
tfidf_matchtitle_stringonly = vectorizer_title.transform(df_all['word_in_title_string_only_string'])
tfidf_matchdescription = vectorizer_description.transform(df_all['word_in_description_string']) 
tfidf_matchdescription_stringonly = vectorizer_description.transform(df_all['word_in_description_string_only_string']) 
tfidf_matchbullets = vectorizer_bullets.transform(df_all['word_in_bullets_string']) 
tfidf_matchbullets_stringonly = vectorizer_bullets.transform(df_all['word_in_bullets_string_only_string']) 

len(vectorizer_title.get_feature_names())
len(vectorizer_description.get_feature_names())
len(vectorizer_bullets.get_feature_names())

uno_title=np.ones((len(vectorizer_title.get_feature_names()),1))
uno_description=np.ones((len(vectorizer_description.get_feature_names()),1))
uno_bullets=np.ones((len(vectorizer_bullets.get_feature_names()),1))

let_title=np.asarray([[len(word)] for word in vectorizer_title.get_feature_names()])
let_description=np.asarray([[len(word)] for word in vectorizer_description.get_feature_names()])
let_bullets=np.asarray([[len(word)] for word in vectorizer_bullets.get_feature_names()])


df_all['tfidf_title_num']=tfidf_title.tocsr().dot(uno_title)
df_all['tfidf_description_num']=tfidf_description.tocsr().dot(uno_description)
df_all['tfidf_bullets_num']=tfidf_bullets.tocsr().dot(uno_bullets)

df_all['tfidf_title_let']=tfidf_title.tocsr().dot(let_title)
df_all['tfidf_description_let']=tfidf_description.tocsr().dot(let_description)
df_all['tfidf_bullets_let']=tfidf_bullets.tocsr().dot(let_bullets)

df_all['tfidf_matchtitle_num']=tfidf_matchtitle.tocsr().dot(uno_title)
df_all['tfidf_matchdescription_num']=tfidf_matchdescription.tocsr().dot(uno_description)
df_all['tfidf_matchbullets_num']=tfidf_matchbullets.tocsr().dot(uno_bullets)

df_all['tfidf_matchtitle_stringonly_num']=tfidf_matchtitle_stringonly.tocsr().dot(uno_title)
df_all['tfidf_matchdescription_stringonly_num']=tfidf_matchdescription_stringonly.tocsr().dot(uno_description)
df_all['tfidf_matchbullets_stringonly_num']=tfidf_matchbullets_stringonly.tocsr().dot(uno_bullets)


df_all['tfidf_title_querythekey_num']=tfidf_title_querythekey.tocsr().dot(uno_title)
df_all['tfidf_title_querybeforethekey_num']=tfidf_title_querybeforethekey.tocsr().dot(uno_title)
df_all['tfidf_title_querythekey_let']=tfidf_title_querythekey.tocsr().dot(let_title)
df_all['tfidf_title_querybeforethekey_let']=tfidf_title_querybeforethekey.tocsr().dot(let_title)



print 'tfidf basic features time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

df_all['tfidf_nn_important_in_title_num']=vectorizer_title.transform(df_all['search_term_tokens'].map(lambda x:nn_important_words(x)) ).tocsr().dot(uno_title)
df_all['tfidf_nn_important_in_description_num']=vectorizer_description.transform(df_all['search_term_tokens'].map(lambda x:nn_important_words(x)) ).tocsr().dot(uno_description)
df_all['tfidf_nn_important_in_bullets_num']=vectorizer_bullets.transform(df_all['search_term_tokens'].map(lambda x:nn_important_words(x)) ).tocsr().dot(uno_bullets)
df_all['tfidf_nn_important_in_title_let']=vectorizer_title.transform(df_all['search_term_tokens'].map(lambda x:nn_important_words(x)) ).tocsr().dot(let_title)
df_all['tfidf_nn_important_in_description_let']=vectorizer_description.transform(df_all['search_term_tokens'].map(lambda x:nn_important_words(x)) ).tocsr().dot(let_description)
df_all['tfidf_nn_important_in_bullets_let']=vectorizer_bullets.transform(df_all['search_term_tokens'].map(lambda x:nn_important_words(x)) ).tocsr().dot(let_bullets)

df_all['tfidf_nn_unimportant_in_title_num']=vectorizer_title.transform(df_all['search_term_tokens'].map(lambda x:nn_unimportant_words(x)) ).tocsr().dot(uno_title)
df_all['tfidf_nn_unimportant_in_description_num']=vectorizer_description.transform(df_all['search_term_tokens'].map(lambda x:nn_unimportant_words(x)) ).tocsr().dot(uno_description)
df_all['tfidf_nn_unimportant_in_bullets_num']=vectorizer_bullets.transform(df_all['search_term_tokens'].map(lambda x:nn_unimportant_words(x)) ).tocsr().dot(uno_bullets)
df_all['tfidf_nn_unimportant_in_title_let']=vectorizer_title.transform(df_all['search_term_tokens'].map(lambda x:nn_unimportant_words(x)) ).tocsr().dot(let_title)
df_all['tfidf_nn_unimportant_in_description_let']=vectorizer_description.transform(df_all['search_term_tokens'].map(lambda x:nn_unimportant_words(x)) ).tocsr().dot(let_description)
df_all['tfidf_nn_unimportant_in_bullets_let']=vectorizer_bullets.transform(df_all['search_term_tokens'].map(lambda x:nn_unimportant_words(x)) ).tocsr().dot(let_bullets)

df_all['tfidf_vbg_in_title_num']=vectorizer_title.transform(df_all['search_term_tokens'].map(lambda x:vbg_words(x)) ).tocsr().dot(uno_title)
df_all['tfidf_vbg_in_description_num']=vectorizer_description.transform(df_all['search_term_tokens'].map(lambda x:vbg_words(x)) ).tocsr().dot(uno_description)
df_all['tfidf_vbg_in_bullets_num']=vectorizer_bullets.transform(df_all['search_term_tokens'].map(lambda x:vbg_words(x)) ).tocsr().dot(uno_bullets)
df_all['tfidf_vbg_in_title_let']=vectorizer_title.transform(df_all['search_term_tokens'].map(lambda x:vbg_words(x)) ).tocsr().dot(let_title)
df_all['tfidf_vbg_in_description_let']=vectorizer_description.transform(df_all['search_term_tokens'].map(lambda x:vbg_words(x)) ).tocsr().dot(let_description)
df_all['tfidf_vbg_in_bullets_let']=vectorizer_bullets.transform(df_all['search_term_tokens'].map(lambda x:vbg_words(x)) ).tocsr().dot(let_bullets)


df_all['tfidf_jj_rb_in_title_num']=vectorizer_title.transform(df_all['search_term_tokens'].map(lambda x:jj_rb_words(x)) ).tocsr().dot(uno_title)
df_all['tfidf_jj_rb_in_description_num']=vectorizer_description.transform(df_all['search_term_tokens'].map(lambda x:jj_rb_words(x)) ).tocsr().dot(uno_description)
df_all['tfidf_jj_rb_in_bullets_num']=vectorizer_bullets.transform(df_all['search_term_tokens'].map(lambda x:jj_rb_words(x)) ).tocsr().dot(uno_bullets)
df_all['tfidf_jj_rb_in_title_let']=vectorizer_title.transform(df_all['search_term_tokens'].map(lambda x:jj_rb_words(x)) ).tocsr().dot(let_title)
df_all['tfidf_jj_rb_in_description_let']=vectorizer_description.transform(df_all['search_term_tokens'].map(lambda x:jj_rb_words(x)) ).tocsr().dot(let_description)
df_all['tfidf_jj_rb_in_bullets_let']=vectorizer_bullets.transform(df_all['search_term_tokens'].map(lambda x:jj_rb_words(x)) ).tocsr().dot(let_bullets)



df_all=df_all.drop(['word_in_title_string','word_in_title_string_only_string','word_in_description_string',\
'word_in_description_string_only_string','word_in_bullets_string','word_in_bullets_string_only_string'],axis=1)

print 'tfidf advanced features time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()





#################################################################
### STEP 8: Unique query, title, etc
#################################################################
"""

aa=list(set(list(df_all['product_description'])))
my_dict={}
for i in range(0,len(set(list(df_all['product_description'])))):
    my_dict[aa[i]]=i
df_all['uniq_product_description']=df_all['product_description'].map(lambda x: 100000+my_dict[x])

aa=list(set(list(df_all['product_title'])))
my_dict={}
for i in range(0,len(set(list(df_all['product_title'])))):
    my_dict[aa[i]]=i
df_all['uniq_product_title']=df_all['product_title'].map(lambda x: 100000+my_dict[x])

aa=list(set(list(df_all['search_term_parsed'])))
my_dict={}
for i in range(0,len(set(list(df_all['search_term_parsed'])))):
    my_dict[aa[i]]=i
df_all['uniq_query']=df_all['search_term_parsed'].map(lambda x: 100000+my_dict[x])



df_all['text']=df_all['uniq_query'].map(lambda x: str(x))+"_"+df_all['uniq_product_title'].map(lambda x: str(x))+"_"+df_all['uniq_product_description'].map(lambda x: str(x))
aa=list(set(list(df_all['text'])))
my_dict={}
for i in range(0,len(set(list(df_all['text'])))):
    my_dict[aa[i]]=i
df_all['uniq_query_title_description']=df_all['text'].map(lambda x: 100000+my_dict[x])

df_all['text']=df_all['uniq_query'].map(lambda x: str(x))+"_"+df_all['uniq_product_title'].map(lambda x: str(x))
aa=list(set(list(df_all['text'])))
my_dict={}
for i in range(0,len(set(list(df_all['text'])))):
    my_dict[aa[i]]=i
df_all['uniq_query_title']=df_all['text'].map(lambda x: 100000+my_dict[x])

df_all['text']=df_all['uniq_query'].map(lambda x: str(x))+"_"+df_all['uniq_product_description'].map(lambda x: str(x))
aa=list(set(list(df_all['text'])))
my_dict={}
for i in range(0,len(set(list(df_all['text'])))):
    my_dict[aa[i]]=i
df_all['uniq_query_description']=df_all['text'].map(lambda x: 100000+my_dict[x])
df_all=df_all.drop(['text'],axis=1)




aa=list(set(list(df_all['search_term_thekey_stemmed'])))
my_dict={}
for i in range(0,len(set(list(df_all['search_term_thekey_stemmed'])))):
    my_dict[aa[i]]=i
df_all['uniq_query_thekey']=df_all['search_term_thekey_stemmed'].map(lambda x: my_dict[x])


aa=list(set(list(df_all['product_title_thekey_stemmed'])))
my_dict={}
for i in range(0,len(set(list(df_all['product_title_thekey_stemmed'])))):
    my_dict[aa[i]]=i
df_all['uniq_product_title_thekey']=df_all['product_title_thekey_stemmed'].map(lambda x: my_dict[x])


aa=list(set(list(df_all['search_term_keys_stemmed'])))
my_dict={}
for i in range(0,len(set(list(df_all['search_term_keys_stemmed'])))):
    my_dict[aa[i]]=i
df_all['uniq_query_keys']=df_all['search_term_keys_stemmed'].map(lambda x: my_dict[x])


aa=list(set(list(df_all['product_title_keys_stemmed'])))
my_dict={}
for i in range(0,len(set(list(df_all['product_title_keys_stemmed'])))):
    my_dict[aa[i]]=i
df_all['uniq_product_title_keys']=df_all['product_title_keys_stemmed'].map(lambda x: my_dict[x])

print 'new unique features time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

df_all['text']=df_all['search_term_keys_stemmed'].map(lambda x: str(x))+"_"+df_all['product_title_keys_stemmed'].map(lambda x: str(x))
aa=list(set(list(df_all['text'])))
my_dict={}
for i in range(0,len(set(list(df_all['text'])))):
    my_dict[aa[i]]=i
df_all['uniq_query_title_keys']=df_all['text'].map(lambda x: 10000+my_dict[x])
df_all=df_all.drop(['text'],axis=1)

df_all['text']=df_all['search_term_thekey_stemmed'].map(lambda x: str(x))+"_"+df_all['product_title_thekey_stemmed'].map(lambda x: str(x))
aa=list(set(list(df_all['text'])))
my_dict={}
for i in range(0,len(set(list(df_all['text'])))):
    my_dict[aa[i]]=i
df_all['uniq_query_title_thekeys']=df_all['text'].map(lambda x: 10000+my_dict[x])
df_all=df_all.drop(['text'],axis=1)


df_all['text']=df_all['search_term_beforethekey_stemmed'].map(lambda x: str(x))+"_"+df_all['search_term_thekey_stemmed'].map(lambda x: str(x))
aa=list(set(list(df_all['text'])))
my_dict={}
for i in range(0,len(set(list(df_all['text'])))):
    my_dict[aa[i]]=i
df_all['uniq_query_beforethekeythekey']=df_all['text'].map(lambda x: 10000+my_dict[x])
df_all=df_all.drop(['text'],axis=1)



df_all['text']=df_all['product_title_beforethekey_stemmed'].map(lambda x: str(x))+"_"+df_all['product_title_thekey_stemmed'].map(lambda x: str(x))
aa=list(set(list(df_all['text'])))
my_dict={}
for i in range(0,len(set(list(df_all['text'])))):
    my_dict[aa[i]]=i
df_all['uniq_title_beforethekeythekey']=df_all['text'].map(lambda x: 10000+my_dict[x])
df_all=df_all.drop(['text'],axis=1)


df_all['text']=df_all['search_term_beforethekey_stemmed'].map(lambda x: str(x))+"_"+df_all['search_term_thekey_stemmed'].map(lambda x: str(x))+"_" \
+df_all['product_title_beforethekey_stemmed'].map(lambda x: str(x))+"_"+df_all['product_title_thekey_stemmed'].map(lambda x: str(x))
aa=list(set(list(df_all['text'])))
my_dict={}
for i in range(0,len(set(list(df_all['text'])))):
    my_dict[aa[i]]=i
df_all['uniq_queryandtitle_beforethekeythekey']=df_all['text'].map(lambda x: 10000+my_dict[x])
df_all=df_all.drop(['text'],axis=1)



aa=list(set(list(df_all['brand_parsed'])))
my_dict={}
for i in range(0,len(set(list(df_all['brand_parsed'])))):
    my_dict[aa[i]]=i
df_all['uniq_brand']=df_all['brand_parsed'].map(lambda x: 10000+my_dict[x])

print 'create unique... variables time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

df_all=df_all.drop(['product_info'],axis=1)


"""

#################################################################
### STEP 9: 'Query expansion'
#################################################################

### For each comination of beforethekey_thekey in query
### create the list of the most common words from product description.
### Since the average relevance tends to be closer to 3 than to 1,
### the majority of matched products are *relevant*. It means that 
### the most common words in matched product description denote 
### high level of relevance. So, we can assess relevance by estimating
### how many common words the prodcut description contains.

t2=time()
df_all['search_term_beforethekey_thekey_stemmed']=df_all['search_term_beforethekey_stemmed']+"_"+df_all['search_term_thekey_stemmed']
aa=list(set(list(df_all['search_term_beforethekey_thekey_stemmed'])))
similarity_dict={}
for i in range(0,len(aa)):
    # get unique words from each product description then concatenate all results:
    all_descriptions= " ".join(list(df_all['product_description_stemmed_woBrand'][df_all['search_term_beforethekey_thekey_stemmed']==aa[i]].map(lambda x: " ".join(list(set(x.split())))   )  ))
    # and transform to a list:    
    all_descriptions_list=all_descriptions.split()
    # vocabulary is simly a set of unique words:
    vocabulary=list(set(all_descriptions_list))
    #count the frequency of each combination of beforethekey_thekey
    cnt=list(df_all['search_term_beforethekey_thekey_stemmed']).count(aa[i])    
    freqs=[1.0*all_descriptions_list.count(w)/cnt for w in vocabulary]
    
    vocabulary+=['dummyword0', 'dummyword1', 'dummyword2', 'dummyword3', 'dummyword4', 'dummyword5',\
    'dummyword6', 'dummyword7', 'dummyword8', 'dummyword9', 'dummyword10', 'dummyword11', 'dummyword12',\
    'dummyword13', 'dummyword14', 'dummyword15', 'dummyword16', 'dummyword17', 'dummyword18', \
    'dummyword19', 'dummyword20', 'dummyword21', 'dummyword22', 'dummyword23', 'dummyword24', 'dummyword25',\
    'dummyword26', 'dummyword27', 'dummyword28', 'dummyword29']
    freqs+=list(np.zeros(30))
    similarity_dict[aa[i]]={"cnt": cnt, 'words':sorted(zip(vocabulary,freqs),key=lambda x: x[1],reverse=True)[0:30]}

    if (i % 2000)==0:
        print ""+str(i)+" out of "+str(len(aa))+" unique combinations; "+str(round((time()-t2)/60,1))+" minutes"

print 'create similarity dict time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

### check the next var for potential overfit
#df_all['frequency_of_beforethekey_thekey']=df_all['search_term_beforethekey_thekey_stemmed'].map(lambda x: int(similarity_dict[x]['cnt']) * int(x!="_"))

df_all['above8_dummy_frequency_of_beforethekey_thekey']=df_all['search_term_beforethekey_thekey_stemmed'].map(lambda x: int(similarity_dict[x]['cnt']>=8 and x!="_"))




### Get the frequency of top words
### Assume the product descriptions consist only of three words:
### word1 appears in 80% of descriptions
### word2 appears in 60% of descriptions
### word3 appears in 40% of descriptions
### Then we encounter a product description with word1 and word3, but no word2.
### In this example the sum frequencies for matched words is 0.8+0.4=1.2;
### the sum of frequencies for all words is 0.8+0.6+0.4=1.8 
def get_description_similarity(str_thekey_pair,str_description,similarity_dict_item):   
    values=np.zeros(30)
    values0=np.zeros(30)
    if str_thekey_pair!="_" and similarity_dict_item['cnt']>=8:
        for j in range(0,30):
            ## frequencies for matched words
            values[j]=similarity_dict_item['words'][j][1]*int(" "+similarity_dict_item['words'][j][0]+" " in " "+str_description+" ")
            #frequencies for all words            
            values0[j]=similarity_dict_item['words'][j][1]
    return sum(values[0:10]),sum(values[0:20]),sum(values[0:30]), sum(values0[0:10]),sum(values0[0:20]),sum(values0[0:30])
            

df_all['description_similarity_tuple']=df_all.apply(lambda x: \
            get_description_similarity(x['search_term_beforethekey_thekey_stemmed'],x['product_description_stemmed'],\
                similarity_dict[x['search_term_beforethekey_thekey_stemmed']]),axis=1)
                

df_all['description_similarity_10']=df_all['description_similarity_tuple'].map(lambda x: x[0]/10.0)
df_all['description_similarity_20']=df_all['description_similarity_tuple'].map(lambda x: x[1]/20.0)
df_all['description_similarity_11-20']=df_all['description_similarity_tuple'].map(lambda x: (x[1]-x[0])/10.0)
df_all['description_similarity_30']=df_all['description_similarity_tuple'].map(lambda x: x[2]/30.0)
df_all['description_similarity_21-30']=df_all['description_similarity_tuple'].map(lambda x: (x[2]-x[1])/10.0)

df_all['description_similarity_10rel']=df_all['description_similarity_tuple'].map(lambda x: 1.0*x[0]/x[3] if x[3]>0.00001 else 0.0)
df_all['description_similarity_20rel']=df_all['description_similarity_tuple'].map(lambda x: 1.0*x[1]/x[4] if x[4]>0.00001 else 0.0)
df_all['description_similarity_11-20rel']=df_all['description_similarity_tuple'].map(lambda x: 1.0*(x[1]-x[0])/(x[4]-x[3]) if (x[4]-x[3])>0.00001 else 0.0)
df_all['description_similarity_30rel']=df_all['description_similarity_tuple'].map(lambda x: 1.0*x[2]/x[5] if x[5]>0.00001 else 0.0)
df_all['description_similarity_21-30rel']=df_all['description_similarity_tuple'].map(lambda x: 1.0*(x[2]-x[1])/(x[5]-x[4]) if (x[5]-x[4])>0.00001 else 0.0)

df_all['description_similarity_11-20to10']=df_all['description_similarity_tuple'].map(lambda x: 1.0*x[4]/x[3] if x[3]>0.00001 else 0.0)
df_all['description_similarity_21-30to10']=df_all['description_similarity_tuple'].map(lambda x: 1.0*x[5]/x[3] if x[3]>0.00001 else 0.0)


# this was added later, so this var is saved separately
df_all['above15_dummy_frequency_of_beforethekey_thekey']=df_all['search_term_beforethekey_thekey_stemmed'].map(lambda x: int(similarity_dict[x]['cnt']>=15 and x!="_"))

t0 = time()
aa=list(set(list(df_all['search_term_beforethekey_thekey_stemmed'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=df_all['description_similarity_20'][df_all['search_term_beforethekey_thekey_stemmed']==aa[i]]
    if i % 200==0:
        print i, "out of", len(aa), ";", round((time()-t0)/60,1) ,'minutes'


def get_percentile_similarity(similarity20, str_thekey_pair, dict_item):   
    output=0.5
    N=len(dict_item)
    if str_thekey_pair!="_" and N>=8:
        n=sum(dict_item>=similarity20)
        assert n>0
        output = 1.0*(n-1)/(N-1)
    return output
    
df_all['description20_percentile']=df_all.apply(lambda x: \
            get_percentile_similarity(x['description_similarity_20'],x['search_term_beforethekey_thekey_stemmed'],\
            my_dict[x['search_term_beforethekey_thekey_stemmed']]),axis=1)


# these vars were added later, so they are saved separately    
df_all[['id','above15_dummy_frequency_of_beforethekey_thekey','description20_percentile']].to_csv(FEATURES_DIR+"/df_feature_above15_ext_wo_google.csv", index=False)
df_all=df_all.drop(['above15_dummy_frequency_of_beforethekey_thekey','description20_percentile'],axis=1)
    




print 'create description similarity variables time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

df_all=df_all.drop(['description_similarity_tuple'],axis=1)

df_all=df_all.drop(['search_term_beforethekey_thekey_stemmed'],axis=1)


df_all.drop(string_variables_list,axis=1).to_csv(FEATURES_DIR+"/df_basic_features_wo_google.csv", index=False)
print 'save file time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()  


#################################################################
### STEP 10: Brand and material dummies
#################################################################

df_all=df_all[['id']+string_variables_list]

brand_dict={}
material_dict={}
import csv
with open(PROCESSINGTEXT_DIR+'/brand_statistics_wo_google.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        brand_dict[row['name']]={'cnt_attribute': int(row['cnt_attribute']), 'cnt_query': int(row['cnt_query'])}
        
with open(PROCESSINGTEXT_DIR+'/material_statistics_wo_google.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        material_dict[row['name']]={'cnt_attribute': int(row['cnt_attribute']), 'cnt_query': int(row['cnt_query'])}

#def create_dummy_value(lst,item):
#    return int(item in lst) * int(len(item)>1)


def col_create_brandmaterial_dummy(col_brandmaterial_txt,bm):
    bm_in_text=col_brandmaterial_txt.map(lambda x: x.split(';'))
    return bm_in_text.map(lambda x: int(bm in x))
    

BM_THRESHOLD=500
cnt=0
cnt1=0
for brand in brand_dict.keys():
    cnt+=1
    if brand_dict[brand]['cnt_attribute']>BM_THRESHOLD:
        cnt1+=1
        df_all["BRANDdummyintitle_"+brand.replace(" ","")]=\
            col_create_brandmaterial_dummy(df_all['brands_in_product_title'],brand)
    if brand_dict[brand]['cnt_query']>BM_THRESHOLD:
        cnt1+=1
        df_all["BRANDdummyinquery_"+brand.replace(" ","")]=\
            col_create_brandmaterial_dummy(df_all['brands_in_search_term'],brand)
    if cnt % 200==0:
        print cnt,"brands processed;", cnt1, "dummies created"

cnt=0
cnt1=0            
for material in material_dict.keys():
    cnt+=1
    if material_dict[material]['cnt_attribute']>BM_THRESHOLD:
        cnt1+=1
        df_all["MATERIALdummyintitle_"+material.replace(" ","")]=\
            col_create_brandmaterial_dummy(df_all['materials_in_product_title'],material)
    if material_dict[material]['cnt_query']>BM_THRESHOLD:
        cnt1+=1
        df_all["MATERIALdummyinquery_"+material.replace(" ","")]=\
            col_create_brandmaterial_dummy(df_all['materials_in_search_term'],material)
    if cnt % 200==0:
        print cnt,"materials processed;", cnt1, "dummies created"    
    

print 'create brand and material dummies time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()        



df_all.drop(string_variables_list,axis=1).to_csv(FEATURES_DIR+"/df_brand_material_dummies_wo_google.csv", index=False)
print 'save file time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()  


#################################################################
### STEP 11: Thekey dummies
#################################################################

df_all=df_all[['id']+string_variables_list]


THEKEY_THRESHOLD=500
lst=list(df_all['product_title_thekey_stemmed'])
freqs=[(w,lst.count(w)) for w in set(lst)]
cnt=0
title_keys=[]
for tpl in sorted(freqs,key=lambda x: x[1],reverse=True):
    if tpl[1]>THEKEY_THRESHOLD:
        cnt+=1
        title_keys.append(tpl[0])
        df_all["THEKEYdummyintitle_"+tpl[0]]=df_all['product_title_thekey_stemmed'].map(lambda x: int(x==tpl[0]))
print cnt, "product_title thekey dummies created"

        
lst=list(df_all['search_term_thekey_stemmed'])
freqs=[(w,lst.count(w)) for w in set(lst)]
cnt=0
query_keys=[]
for tpl in sorted(freqs,key=lambda x: x[1],reverse=True):
    if tpl[1]>THEKEY_THRESHOLD:
        cnt+=1
        query_keys.append(tpl[0])
        df_all["THEKEYdummyinquery_"+tpl[0]]=df_all['search_term_thekey_stemmed'].map(lambda x: int(x==tpl[0]))
print cnt, "search_term thekey dummies created"

BEFORETHEKEYTHEKEY_THRESHOLD=300
lst=list(df_all['product_title_beforethekey_stemmed']+"_"+df_all['product_title_thekey_stemmed'])
freqs=[(w,lst.count(w)) for w in set(lst)]
cnt=0
for tpl in sorted(freqs,key=lambda x: x[1],reverse=True):
    if tpl[1]>BEFORETHEKEYTHEKEY_THRESHOLD:
        cnt+=1
        df_all["BTK_TKdummyintitle_"+tpl[0]]=(df_all['product_title_beforethekey_stemmed']+"_"+df_all['product_title_thekey_stemmed']).map(lambda x: int(x==tpl[0]))
print cnt, "product_title beforethekey_thekey dummies created"   

lst=list(df_all['search_term_beforethekey_stemmed']+"_"+df_all['search_term_thekey_stemmed'])
freqs=[(w,lst.count(w)) for w in set(lst)]
cnt=0
for tpl in sorted(freqs,key=lambda x: x[1],reverse=True):
    if tpl[1]>BEFORETHEKEYTHEKEY_THRESHOLD:
        cnt+=1
        df_all["BTK_TKdummyinquery_"+tpl[0]]=(df_all['search_term_beforethekey_stemmed']+"_"+df_all['search_term_thekey_stemmed']).map(lambda x: int(x==tpl[0]))
print cnt, "search_term beforethekey_thekey dummies created"


print 'create thekeys dummies time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()        
       
       
df_all.drop(string_variables_list,axis=1).to_csv(FEATURES_DIR+"/df_thekey_dummies_wo_google.csv", index=False)
print 'save file time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()         
       
print 'TOTAL FEATURE EXTRACTION TIME:',round((time()-t1)/60,1) ,'minutes\n'
    


################################


