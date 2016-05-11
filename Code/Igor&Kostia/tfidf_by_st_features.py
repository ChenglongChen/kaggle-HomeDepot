# -*- coding: utf-8 -*-
"""
Code for calculating some similar to TFIDF features for all docs related to the same unique search term.
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




#data loading
df_all=pd.read_csv(PROCESSINGTEXT_DIR+"/df_train_and_test_processed.csv", encoding="ISO-8859-1")
df_all1=pd.read_csv(PROCESSINGTEXT_DIR+"/df_product_descriptions_processed.csv", encoding="ISO-8859-1")
df_all2 = pd.merge(df_all, df_all1, how="left", on="product_uid")
df_all = df_all2


p = df_all.keys()
for i in range(len(p)):
    print p[i]

for var in df_all.keys():
    df_all[var]=df_all[var].fillna("")

#building base of the documents related to unique search term
st = df_all["search_term_stemmed"]
pt = df_all["product_title_stemmed"]
pd = df_all["product_description_stemmed"]

st_l = list(st)
st_lu = list(set(st))

t=list()
another_t=list()
for i in range(len(st_lu)):
    t.append("")
    another_t.append("")

   
    
for i in range(len(st_lu)):
    
    another_t[i] = list(pt[st==st_lu[i]])
    t[i]= list(pd[st==st_lu[i]])
    


 
#damareu-levenstein distance 
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
   
#make a list of words which are not close in dld distance meaning
def dld_replacer(s1):
    s=s1
    uwords=list()
    allwords=list()
    for i in range(len(s)):
        words = s[i].split(" ")
        for j in range(len(words)):
           for i1 in range(len(s)):
               words2 = s[i1].split(" ")
               for j1 in range(len(words2)): 
                  # print i,j,i1,j1,len(s)
                   if dld(words[j], words2[j1])<min(3,max(len(words2[j1])-3,1)):
                       allwords.append(words2[j1])
                       words2[j1]=words[j]
                       uwords.append(words[j])
                       
               d=""
               #print d
               for t in range(len(words2)):
                   d=d+words2[t]+" "
                   #print d               
              # print d
               d=d[0:len(d)-1]
               s[i1]=d    
           d=""        
           for t1 in range(len(words)):
                   d=d+words[t1]+" "
           #print d
           d=d[0:len(d)-1]
           s[i]=d
    allwords = list(set(allwords))      
    uwords=list(set(uwords))               
    return s, allwords, uwords

#make a simle list of word
def simple_replacer(s1):
    s=s1
    uwords=list()
    for i in range(len(s)):
        words = s[i].split(" ")
        for j in range(len(words)):
              uwords.append(words[j])
    uwords=list(set(uwords))               
    return uwords


new_t=t
new_t1=list()
alwords1=list()
uwords=list()
uwords1=list()


for i in range(len(t)):
    tmp1,tmp2,tmp3= dld_replacer(another_t[i])
    uwords1.append(tmp3)
    new_t1.append(tmp1)
    alwords1.append(tmp2)
    tmp4= simple_replacer(t[i])
    uwords.append(tmp4)
    

idf_list=list()
t_length=list()
d_length=list()

idf_list1=list()
t_length1=list()
d_length1=list()

#calculate count of unique words for each document related to unique search term
def idf_from_list(key_word,new_t,uwords):
    j = st_lu.index(key_word)
    idf_list=list()
    d = 0
    l = len(new_t[j])
    
    for t in range(l):
        tmp_list=list()
        words=new_t[j][t].split(" ")
        for i in range(len(uwords[j])):
        
            d = d + words.count(uwords[j][i])
            tmp_list.append(words.count(uwords[j][i]))
        idf_list.append(tmp_list)
        
    return  idf_list, d, l   
        
for i in  range(len(st_lu)) :   
    tmp1,tmp2,tmp3 = idf_from_list(st_lu[i],new_t,uwords)
    idf_list.append(tmp1)
    d_length.append(tmp2)
    t_length.append(tmp3)
    
    tmp1,tmp2,tmp3 = idf_from_list(st_lu[i],new_t1,uwords1)
    idf_list1.append(tmp1)
    d_length1.append(tmp2)
    t_length1.append(tmp3)
    
    if (i%1000)==0:
        print i


st = df_all["search_term_stemmed"]
pt = df_all["product_title_stemmed"]
pd = df_all["product_description_stemmed"]


   
    
for i in range(len(st_lu)):
    another_t[i] = list(pt[st==st_lu[i]])
    t[i]= list(pd[st==st_lu[i]])
    if (i%1000)==0:
        print i


list1=list()
list2=list()
list3=list()
list4=list()
list5=list()
list6=list()
list7=list()
list8=list()
list9=list()
list10=list()
list11=list()
list12=list()


#calculate features using st=search_term and pd=product_description

for i in range(len(df_all)):
    df_parsed=pd
    #j =  st_lu.index(df_all["search_term_parsed"][i])
    #k=t[j].index(df_all["product_title_parsed"][i])
    j =  st_lu.index(df_all["search_term_stemmed"][i])
    k=t[j].index(df_parsed[i])
    if d_length[j]==0:
        d_length[j]=1
    f1=(sum(idf_list[j][k])+0.0)/d_length[j]
    f2=(sum(idf_list[j][k])+0.0)/t_length[j]
    f3=d_length[j]
    f4=t_length[j]
    #f5=len(df_all["product_title_parsed"][i].split(" "))/len(list(set(new_t[j][k].split(" "))))
    f5=len(df_parsed[i].split(" "))/len(list(set(new_t[j][k].split(" "))))
    f6=(sum(idf_list[j][k])+0.0)*m.log(d_length[j])
    f7=(sum(idf_list[j][k])+0.0)*m.log(t_length[j])
    f8=(sum(idf_list[j][k])+0.0)/((d_length[j]+0.0)/len(idf_list[j]))
    f9=(sum(idf_list[j][k])+0.0)/((t_length[j]+0.0)/len(idf_list[j]))
    f10=(len(list(set(list(st_lu[j].split(" "))))) +0.0)/d_length[j]
    f11=(len(list(set(list(st_lu[j].split(" "))))) +0.0)/t_length[j]
    f12=len(list(set(list(st_lu[j].split(" "))))) /((d_length[j]+0.0)/len(idf_list[j]))
    if (i%1000)==0:
        print i

    list1.append(f1)
    list2.append(f2)
    list3.append(f3)
    list4.append(f4)
    list5.append(f5)
    list6.append(f6)
    list7.append(f7)
    list8.append(f8)
    list9.append(f9)
    list10.append(f10)
    list11.append(f11)
    list12.append(f12)

list_of_list=[list1,list2,list3,list4,list5,list6,list7,list8,list9,list10,list11,list12]
st_names=["id"]    
for j in range(12):    
    df_all["st_tfidf_"+str(j)]=list_of_list[j]
    st_names.append("st_tfidf_"+str(j))
        

list1=list()
list2=list()
list3=list()
list4=list()
list5=list()
list6=list()
list7=list()
list8=list()
list9=list()
list10=list()
list11=list()
list12=list()



new_t=new_t1
idf_list=idf_list1
d_length=d_length1
t_length=t_length1
t=another_t


#calculate features using st=search_term and pd=product_title

for i in range(len(df_all)):
    df_parsed=pt
    #j =  st_lu.index(df_all["search_term_parsed"][i])
    #k=t[j].index(df_all["product_title_parsed"][i])
    j =  st_lu.index(df_all["search_term_stemmed"][i])
    k=t[j].index(df_parsed[i])
    if d_length[j]==0:
        d_length[j]=1   
    f1=(sum(idf_list[j][k])+0.0)/d_length[j]
    f2=(sum(idf_list[j][k])+0.0)/t_length[j]
    f3=d_length[j]
    f4=t_length[j]
    #f5=len(df_all["product_title_parsed"][i].split(" "))/len(list(set(new_t[j][k].split(" "))))
    f5=len(df_parsed[i].split(" "))/len(list(set(new_t[j][k].split(" "))))
    f6=(sum(idf_list[j][k])+0.0)*m.log(d_length[j])
    f7=(sum(idf_list[j][k])+0.0)*m.log(t_length[j])
    f8=(sum(idf_list[j][k])+0.0)/((d_length[j]+0.0)/len(idf_list[j]))
    f9=(sum(idf_list[j][k])+0.0)/((t_length[j]+0.0)/len(idf_list[j]))
    f10=(len(list(set(list(st_lu[j].split(" "))))) +0.0)/d_length[j]
    f11=(len(list(set(list(st_lu[j].split(" "))))) +0.0)/t_length[j]
    f12=len(list(set(list(st_lu[j].split(" "))))) /((d_length[j]+0.0)/len(idf_list[j]))
    if (i%1000)==0:
        print i

    list1.append(f1)
    list2.append(f2)
    list3.append(f3)
    list4.append(f4)
    list5.append(f5)
    list6.append(f6)
    list7.append(f7)
    list8.append(f8)
    list9.append(f9)
    list10.append(f10)
    list11.append(f11)
    list12.append(f12)


for j in range(12):    
    df_all["st_tfidf_"+str(j)"+".1"]=list_of_list[j]
    st_names.append("st_tfidf_"+str(j)"+".1")

#save features
b=df_all[st_names]
b.to_csv(FEATURES_DIR+"/df_st_tfidf.csv", index=False) 

