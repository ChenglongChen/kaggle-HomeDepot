# -*- coding: utf-8 -*-
"""
Initial text preprocessing.
Although text processing can be technically done within feature generation functions, 
we found it to be very efficient to make all preprocessing first and only then move to 
feature generation. It is because the same processed text is used as an input to
generate several different features. 

This file is the same as text_processing.py, except this line is commented
df_all['search_term']=df_all['search_term'].map(lambda x: google_dict[x] if x in google_dict.keys() else x)   
and the ouput is saved with '_wo_google' added to file names.

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
stoplist.append('till')  # add 'till' to stoplist

# 'can' also might mean 'a container' like in 'trash can' 
# so we create a separate stop list without 'can' to be used for query and product title
stoplist_wo_can=stoplist[:]
stoplist_wo_can.remove('can')


from homedepot_functions import *
from google_dict import *

t0 = time()
t1 = time()


############################################
####### PREPROCESSING ######################
############################################

### load train and test ###################
df_train = pd.read_csv(DATA_DIR+'/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv(DATA_DIR+'/test.csv', encoding="ISO-8859-1")
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

### load product attributes ###############
df_attr = pd.read_csv(DATA_DIR+'/attributes.csv', encoding="ISO-8859-1")

print 'loading time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


### find unique brands from the attributes file
### for a few product_uids there are at least two different names in "MFG Brand Name"
### in such cases we keep only one of the names
df_all = pd.merge(df_all, df_attr[df_attr['name']=="MFG Brand Name"][['product_uid','value']], how='left', on='product_uid')
df_all['brand']=df_all['value'].fillna("").map(lambda x: x.encode('utf-8'))
df_all=df_all.drop('value',axis=1)


### Create a list of words with lowercase and uppercase letters 
### Examples: 'InSinkErator', 'EpoxyShield'
### They are words from brand names or words from product title.
### The dict is used to correct product description which contins concatenated 
### lines of text without separators : 
### ---View lawn edgings and brick/ paver edgingsUtility stakes can be used for many purposes---
### Here we need to replace 'edgingsUtility' with 'edgings utility'. 
### But we don't need to replace 'InSinkErator' with 'in sink erator'
add_space_stop_list=[]
uniq_brands=list(set(list(df_all['brand'])))
for i in range(0,len(uniq_brands)):
    uniq_brands[i]=simple_parser(uniq_brands[i])
    if re.search(r'[a-z][A-Z][a-z]',uniq_brands[i])!=None:
        for word in uniq_brands[i].split():
            if re.search(r'[a-z][A-Z][a-z]',word)!=None:
                add_space_stop_list.append(word.lower())
add_space_stop_list=list(set(add_space_stop_list))      
print len(add_space_stop_list)," words from brands in add_space_stop_list"
                
uniq_titles=list(set(list(df_all['product_title'])))
for i in range(0,len(uniq_titles)):
    uniq_titles[i]=simple_parser(uniq_titles[i])
    if re.search(r'[a-z][A-Z][a-z]',uniq_titles[i])!=None:
        for word in uniq_titles[i].split():
            if re.search(r'[a-z][A-Z][a-z]',word)!=None:
                add_space_stop_list.append(word.lower())    
add_space_stop_list=list(set(add_space_stop_list))      
print len(add_space_stop_list) ," total words from brands and product titles in add_space_stop_list\n"
                

#################################################################
##### First step of spell correction: using the Google dict
##### from the forum
# https://www.kaggle.com/steubk/home-depot-product-search-relevance/fixing-typos

## the following line is commented in this file
####df_all['search_term']=df_all['search_term'].map(lambda x: google_dict[x] if x in google_dict.keys() else x)   




#################################################################
##### AUTOMATIC SPELL CHECKER ###################################
#################################################################

### A simple spell checker is implemented here
### First, we get unique words from search_term and product_title
### Then, we count how many times word occurs in search_term and product_title
### Finally, if the word is not present in product_title and not meaningful
### (i.e. wn.synsets(word) returns empty list), the word is likely 
### to be misspelled, so we try to correct it using bigrams, words from matched
### products or all products. The best match is chosen using 
### difflib.SequenceMatcher()


def is_word_in_string(word,s):
    return word in s.split() 
    
def create_bigrams(s):
    lst=[word for word in s.split() if len(re.sub('[^0-9]', '', word))==0 and len(word)>2]
    output=""
    i=0
    if len(lst)>=2:
        while i<len(lst)-1:
            output+= " "+lst[i]+"_"+lst[i+1]
            i+=1
    return output


df_all['product_title_simpleparsed']=df_all['product_title'].map(lambda x: simple_parser(x).lower())
df_all['search_term_simpleparsed']=df_all['search_term'].map(lambda x: simple_parser(x).lower())

str_title=" ".join(list(df_all['product_title'].map(lambda x: simple_parser(x).lower())))
str_query=" ".join(list(df_all['search_term'].map(lambda x: simple_parser(x).lower())))

# create bigrams
bigrams_str_title=" ".join(list(df_all['product_title'].map(lambda x: create_bigrams(simple_parser(x).lower()))))
bigrams_set=set(bigrams_str_title.split())

### count word frequencies for query and product title
my_dict={}
str1= str_title+" "+str_query
for word in list(set(list(str1.split()))):
    my_dict[word]={"title":0, "query":0, 'word':word}
for word in str_title.split():
    my_dict[word]["title"]+=1    
for word in str_query.split():
    my_dict[word]["query"]+=1


### 1. Process words without digits
### Potential errors: words that appear only in query
### Correct words: 5 or more times in product_title
errors_dict={}
correct_dict={}
for word in my_dict.keys():
    if len(word)>=3 and len(re.sub('[^0-9]', '', word))==0:
        if my_dict[word]["title"]==0:
            if len(wn.synsets(word))>0 \
            or (word.endswith('s') and  (word[:-1] in my_dict.keys()) and my_dict[word[:-1]]["title"]>0)\
            or (word[-1]!='s' and (word+'s' in my_dict.keys()) and my_dict[word+'s']["title"]>0):
                1
            else:
                errors_dict[word]=my_dict[word]
        elif my_dict[word]["title"]>=5:
            correct_dict[word]=my_dict[word]


### for each error word try finding a good match in bigrams, matched products, all products
cnt=0
NN=len(errors_dict.keys())
t0=time()
for i in range(0,len(errors_dict.keys())):
    word=sorted(errors_dict.keys())[i]
    cnt+=1
    lst=[]
    lst_tuple=[]
    suggested=False
    suggested_word=""
    rt_max=0
    
    # if only one word in query, use be more selective in choosing a correction
    min_query_len=min(df_all['search_term_simpleparsed'][df_all['search_term_simpleparsed'].map(lambda x: is_word_in_string(word,x))].map(lambda x: len(x.split())))
    delta=0.05*int(min_query_len<2)
    
    words_from_matched_titles=[item for item in \
        " ".join(list(set(df_all['product_title_simpleparsed'][df_all['search_term_simpleparsed'].map(lambda x: is_word_in_string(word,x))]))).split() \
        if len(item)>2 and len(re.sub('[^0-9]', '', item))==0]
    words_from_matched_titles=list(set(words_from_matched_titles))
    words_from_matched_titles.sort()
    
    source=""
    for bigram in bigrams_set:
        if bigram.replace("_","")==word:
            suggested=True
            suggested_word=bigram.replace("_"," ")
            source="from bigrams"
            
    if source=="":
        for correct_word in words_from_matched_titles: 
            rt, rt_scaled = seq_matcher(word,correct_word)
            #print correct_word, rt,rt_scaled
            
            if rt>0.75+delta or (len(word)<6 and rt>0.68+delta):
                lst.append(correct_word)
                lst_tuple.append((correct_word,my_dict[correct_word]["title"]))
                if rt>rt_max:
                    rt_max=rt
                    suggested=True
                    source="from matched products"
                    suggested_word=correct_word
                elif rt==rt_max and seq_matcher("".join(sorted(word)),"".join(sorted(correct_word)))[0]>seq_matcher("".join(sorted(word)),"".join(sorted(suggested_word)))[0]:
                    suggested_word=correct_word
                elif rt==rt_max:
                    suggested=False
                    source=""
        
    if source=="" and len(lst)==0:
        source="from all products"
        for correct_word in correct_dict.keys():
            rt, rt_scaled = seq_matcher(word,correct_word)
            #print correct_word, rt,rt_scaled
            if correct_dict[correct_word]["title"]>10 and (rt>0.8+delta or (len(word)<6 and rt>0.73+delta)):
                #print correct_word, rt,rt_scaled
                lst.append(correct_word)
                lst_tuple.append((correct_word,correct_dict[correct_word]["title"]))
                if rt>rt_max:
                    rt_max=rt
                    suggested=True
                    suggested_word=correct_word
                elif rt==rt_max and seq_matcher("".join(sorted(word)),"".join(sorted(correct_word)))[0]>seq_matcher("".join(sorted(word)),"".join(sorted(suggested_word)))[0]:
                    suggested_word=correct_word
                elif rt==rt_max: 
                    suggested=False

    if suggested==True:
        errors_dict[word]["suggestion"]=suggested_word
        errors_dict[word]["others"]=lst_tuple
        errors_dict[word]["source"]=source
    else:
        errors_dict[word]["suggestion"]=""
        errors_dict[word]["others"]=lst_tuple
        errors_dict[word]["source"]=source
    #print(cnt, word, errors_dict[word]["query"], errors_dict[word]["suggestion"], source, errors_dict[word]["others"])
    #if (cnt % 20)==0:
    #    print cnt, " out of ", NN, "; ", round((time()-t0),1) ,' sec'

### 2. Add some words with digits
### If the word begins with a meanigful part [len(wn.synsets(srch.group(0)))>0],
### ends with a number and has vowels
for word in my_dict.keys():
    if my_dict[word]['query']>0 and my_dict[word]['title']==0 \
    and len(re.sub('[^0-9]', '', word))!=0 and len(re.sub('[^a-z]', '', word))!=0:
        srch=re.search(r'(?<=^)[a-z][a-z][a-z]+(?=[0-9])',word)
        if srch!=None and len(wn.synsets(srch.group(0)))>0 \
        and len(re.sub('[^aeiou]', '', word))>0 and word[-1] in '0123456789': 
            errors_dict[word]=my_dict[word]
            errors_dict[word]["source"]="added space before digit"
            errors_dict[word]["suggestion"]=re.sub(r'(?<=^)'+srch.group(0)+r'(?=[a-zA-Z0-9])',srch.group(0)+' ',word)
            #print word, re.sub(r'(?<=^)'+srch.group(0)+r'(?=[a-zA-Z0-9])',srch.group(0)+' ',word)

### save dictionary
corrections_df=pd.DataFrame(errors_dict).transpose()
corrections_df.to_csv(PROCESSINGTEXT_DIR+"/automatically_generated_word_corrections_wo_google.csv")

print 'building spell checker time:',round((time()-t0)/60,1) ,'minutes\n'

##### END OF SPELL CHECKER ######################################
#################################################################


########################################
##### load words for spell checker
spell_check_dict={}
for word in errors_dict.keys():
    if errors_dict[word]['suggestion']!="":
        spell_check_dict[word]=errors_dict[word]['suggestion']

"""
spell_check_dict={}
with open(PROCESSINGTEXT_DIR+"/automatically_generated_word_corrections_wo_google.csv") as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
         if row['suggestion']!="":
             spell_check_dict[row['word']]=row['suggestion']
"""





###############################################
### parse query and product title
df_all['search_term_parsed']=col_parser(df_all['search_term'],automatic_spell_check_dict=spell_check_dict,\
            add_space_stop_list=[]).map(lambda x: x.encode('utf-8'))
df_all['search_term_parsed_wospellcheck']=col_parser(df_all['search_term'],automatic_spell_check_dict={},\
            add_space_stop_list=[]).map(lambda x: x.encode('utf-8'))
print 'search_term parsing time:',round((time()-t0)/60,1) ,'minutes\n'



t0 = time()

### function to check whether queries parsed with and without spell correction are identical
def match_queries(q1,q2):
    q1=re.sub('[^a-z\ ]', '', q1)
    q2=re.sub('[^a-z\ ]', '', q2)
    q1= " ".join([word[0:(len(word)-int(word[-1]=='s'))] for word in q1.split()])
    q2= " ".join([word[0:(len(word)-int(word[-1]=='s'))] for word in q2.split()])
    return difflib.SequenceMatcher(None, q1,q2).ratio()


df_all['is_query_misspelled']=df_all.apply(lambda x: \
            match_queries(x['search_term_parsed'],x['search_term_parsed_wospellcheck']),axis=1)
df_all=df_all.drop(['search_term_parsed_wospellcheck'],axis=1)    
print 'create dummy "is_query_misspelled" time:',round((time()-t0)/60,1) ,'minutes\n'



t0 = time()
df_all['product_title_parsed']=col_parser(df_all['product_title'],add_space_stop_list=[],\
                remove_from_brackets=True).map(lambda x: x.encode('utf-8'))
print 'product_title parsing time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()



#################################################################
##### COUNT BRAND NAMES #########################################
#################################################################

### some brand names in "MFG Brand Name" of attributes.csv have a few words
### but it is much more likely for a person to search for brand 'BEHR' 
### than 'BEHR PREMIUM PLUS ULTRA'. That is why we replace long brand names 
### with a shorter alternatives 
replace_brand_dict={
'acurio latticeworks': 'acurio', 
'american kennel club':'akc',
'amerimax home products': 'amerimax',
'barclay products':'barclay',
'behr marquee': 'behr', 
'behr premium': 'behr', 
'behr premium deckover': 'behr', 
'behr premium plus': 'behr', 
'behr premium plus ultra': 'behr', 
'behr premium textured deckover': 'behr', 
'behr pro': 'behr', 
'bel air lighting': 'bel air',
'bootz industries':'bootz',
'campbell hausfeld':'campbell',
'columbia forest products': 'columbia',
'essick air products':'essick air',
'evergreen enterprises':'evergreen',
'feather river doors': 'feather river', 
'gardner bender':'gardner',
'ge parts':'ge',
'ge reveal':'ge',
'gibraltar building products':'gibraltar',
'gibraltar mailboxes':'gibraltar',
'glacier bay':'glacier',
'great outdoors by minka lavery': 'great outdoors', 
'hamilton beach': 'hamilton',
'hampton bay':'hampton',
'hampton bay quickship':'hampton',
'handy home products':'handy home',
'hickory hardware': 'hickory', 
'home accents holiday': 'home accents',
'home decorators collection': 'home decorators',
'homewerks worldwide':'homewerks',
'klein tools': 'klein',
'lakewood cabinets':'lakewood',
'leatherman tool group':'leatherman',
'legrand adorne':'legrand',
'legrand wiremold':'legrand',
'lg hausys hi macs':'lg',
'lg hausys viatera':'lg',
'liberty foundry':'liberty',
'liberty garden':'liberty',
'lithonia lighting':'lithonia',
'loloi rugs':'loloi',
'maasdam powr lift':'maasdam',
'maasdam powr pull':'maasdam',
'martha stewart living': 'martha stewart',
'merola tile': 'merola',
'miracle gro':'miracle',
'miracle sealants':'miracle',
'mohawk home': 'mohawk',
'mtd genuine factory parts':'mtd',
'mueller streamline': 'mueller',
'newport coastal': 'newport',
'nourison overstock':'nourison',
'nourison rug boutique':'nourison',
'owens corning': 'owens', 
'premier copper products':'premier',
'price pfister':'pfister',
'pride garden products':'pride garden',
'prime line products':'prime line',
'redi base':'redi',
'redi drain':'redi',
'redi flash':'redi',
'redi ledge':'redi',
'redi neo':'redi',
'redi niche':'redi',
'redi shade':'redi',
'redi trench':'redi',
'reese towpower':'reese',
'rheem performance': 'rheem',
'rheem ecosense': 'rheem',
'rheem performance plus': 'rheem',
'rheem protech': 'rheem',
'richelieu hardware':'richelieu',
'rubbermaid commercial products': 'rubbermaid', 
'rust oleum american accents': 'rust oleum', 
'rust oleum automotive': 'rust oleum', 
'rust oleum concrete stain': 'rust oleum', 
'rust oleum epoxyshield': 'rust oleum', 
'rust oleum flexidip': 'rust oleum', 
'rust oleum marine': 'rust oleum', 
'rust oleum neverwet': 'rust oleum', 
'rust oleum parks': 'rust oleum', 
'rust oleum professional': 'rust oleum', 
'rust oleum restore': 'rust oleum', 
'rust oleum rocksolid': 'rust oleum', 
'rust oleum specialty': 'rust oleum', 
'rust oleum stops rust': 'rust oleum', 
'rust oleum transformations': 'rust oleum', 
'rust oleum universal': 'rust oleum', 
'rust oleum painter touch 2': 'rust oleum',
'rust oleum industrial choice':'rust oleum',
'rust oleum okon':'rust oleum',
'rust oleum painter touch':'rust oleum',
'rust oleum painter touch 2':'rust oleum',
'rust oleum porch and floor':'rust oleum',
'salsbury industries':'salsbury',
'simpson strong tie': 'simpson', 
'speedi boot': 'speedi', 
'speedi collar': 'speedi', 
'speedi grille': 'speedi', 
'speedi products': 'speedi', 
'speedi vent': 'speedi', 
'pass and seymour': 'seymour',
'pavestone rumblestone': 'rumblestone',
'philips advance':'philips',
'philips fastener':'philips',
'philips ii plus':'philips',
'philips manufacturing company':'philips',
'safety first':'safety 1st',
'sea gull lighting': 'sea gull',
'scott':'scotts',
'scotts earthgro':'scotts',
'south shore furniture': 'south shore', 
'tafco windows': 'tafco',
'trafficmaster allure': 'trafficmaster', 
'trafficmaster allure plus': 'trafficmaster', 
'trafficmaster allure ultra': 'trafficmaster', 
'trafficmaster ceramica': 'trafficmaster', 
'trafficmaster interlock': 'trafficmaster', 
'thomas lighting': 'thomas', 
'unique home designs':'unique home',
'veranda hp':'veranda',
'whitehaus collection':'whitehaus',
'woodgrain distritubtion':'woodgrain',
'woodgrain millwork': 'woodgrain', 
'woodford manufacturing company': 'woodford', 
'wyndham collection':'wyndham',
'yardgard select': 'yardgard',
'yosemite home decor': 'yosemite'
}

df_all['brand_parsed']=col_parser(df_all['brand'].map(lambda x: re.sub('^[t|T]he ', '', x.replace(".N/A","").replace("N.A.","").replace("n/a","").replace("Generic Unbranded","").replace("Unbranded","").replace("Generic",""))),add_space_stop_list=add_space_stop_list)
list_brands=list(df_all['brand_parsed'])

df_all['brand_parsed']=df_all['brand_parsed'].map(lambda x: replace_brand_dict[x] if x in replace_brand_dict.keys() else x)



### count frequencies of brands in query and product_title
str_query=" : ".join(list(df_all['search_term_parsed'])).lower()
print "\nGenerating brand dict: How many times each brand appears in the dataset?"
brand_dict=get_attribute_dict(list_brands,str_query=str_query)

### These words are likely to mean other things than brand names. 
### For example, it would not be prudent to consider each occurence of 'design' or 'veranda' as a brand name.
### We decide not to use these words as brands and exclude them from our brand dictionary.
# The list is shared on the forum.
del_list=['aaa','off','impact','square','shelves','finish','ring','flood','dual','ball','cutter',\
'max','off','mat','allure','diamond','drive', 'edge','anchor','walls','universal','cat', 'dawn','ion','daylight',\
'roman', 'weed eater', 'restore', 'design', 'caddy', 'pole caddy', 'jet', 'classic', 'element', 'aqua',\
'terra', 'decora', 'ez', 'briggs', 'wedge', 'sunbrella',  'adorne', 'santa', 'bella', 'duck', 'hotpoint',\
'duck', 'tech', 'titan', 'powerwasher', 'cooper lighting', 'heritage', 'imperial', 'monster', 'peak', 
'bell', 'drive', 'trademark', 'toto', 'champion', 'shop vac', 'lava', 'jet', 'flood', \
'roman', 'duck', 'magic', 'allen', 'bunn', 'element', 'international', 'larson', 'tiki', 'titan', \
 'space saver', 'cutter', 'scotch', 'adorne', 'ball', 'sunbeam', 'fatmax', 'poulan', 'ring', 'sparkle', 'bissell', \
 'universal', 'paw', 'wedge', 'restore', 'daylight', 'edge', 'americana', 'wacker', 'cat', 'allure', 'bonnie plants', \
 'troy', 'impact', 'buffalo', 'adams', 'jasco', 'rapid dry', 'aaa', 'pole caddy', 'pac', 'seymour', 'mobil', \
 'mastercool', 'coca cola', 'timberline', 'classic', 'caddy', 'sentry', 'terrain', 'nautilus', 'precision', \
 'artisan', 'mural', 'game', 'royal', 'use', 'dawn', 'task', 'american line', 'sawtrax', 'solo', 'elements', \
 'summit', 'anchor', 'off', 'spruce', 'medina', 'shoulder dolly', 'brentwood', 'alex', 'wilkins', 'natural magic', \
 'kodiak', 'metro', 'shelter', 'centipede', 'imperial', 'cooper lighting', 'exide', 'bella', 'ez', 'decora', \
 'terra', 'design', 'diamond', 'mat', 'finish', 'tilex', 'rhino', 'crock pot', 'legend', 'leatherman', 'remove', \
 'architect series', 'greased lightning', 'castle', 'spirit', 'corian', 'peak', 'monster', 'heritage', 'powerwasher',\
 'reese', 'tech', 'santa', 'briggs', 'aqua', 'weed eater', 'ion', 'walls', 'max', 'dual', 'shelves', 'square',\
 'hickory', "vikrell", "e3", "pro series", "keeper", "coastal shower doors", 'cadet','church','gerber','glidden',\
 'cooper wiring devices', 'border blocks', 'commercial electric', 'pri','exteria','extreme', 'veranda',\
 'gorilla glue','gorilla','shark','wen']
del_list=list(set(list(del_list)))

for key in del_list:
    if key in brand_dict.keys():
        del(brand_dict[key])

# save to file
brand_df=pd.DataFrame(brand_dict).transpose()
brand_df.to_csv(PROCESSINGTEXT_DIR+"/brand_statistics_wo_google.csv")


"""
brand_dict={}
import csv
with open(PROCESSINGTEXT_DIR+"/brand_statistics_wo_google.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        brand_dict[row['name']]={'cnt_attribute': int(row['cnt_attribute']), 'cnt_query': int(row['cnt_query']), 
        'name': row['name'], 'nwords': int(row['nwords'])}
"""


### Later we will create features like match between brands in query and product titles.
### But we only process brands that apper frequently enough in the dataset:
### Either 8+ times in product title or [1+ time in query and 3+ times in product title]
for item in brand_dict.keys():
    if (brand_dict[item]['cnt_attribute']>=3 and brand_dict[item]['cnt_query']>=1) \
    or (brand_dict[item]['cnt_attribute'])>=8:
        1
    else:
        del(brand_dict[item])

brand_df=pd.DataFrame(brand_dict).transpose().sort(['cnt_query'], ascending=[1])
print 'brand dict creation time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


##### END OF COUNT BRAND NAMES ##################################
#################################################################


#################################################################
##### COUNT MATERIALS ###########################################
#################################################################

### First, create list of unique materials. We need to replace some complex names
### (see change_material() function)
### File attributes.csv for some product_uid contains multiple different values of "Material"
### That is why we have to concatenate all such values to ensure that each product_uid
### has only one value for material
tmp_material=df_attr[df_attr['name']=="Material"][['product_uid','value']]
tmp_material=tmp_material[tmp_material['value']!="Other"]
tmp_material=tmp_material[tmp_material['value']!="*"]
def change_material(s):
    replace_dict={'Medium Density Fiberboard (MDF)':'mdf', 'High Density Fiberboard (HDF)':'hdf',\
    'Fibre Reinforced Polymer (FRP)': 'frp', 'Acrylonitrile Butadiene Styrene (ABS)': 'abs',\
    'Cross-Linked Polyethylene (PEX)':'pex', 'Chlorinated Poly Vinyl Chloride (CPVC)': 'cpvc',\
    'PVC (vinyl)': 'pvc','Thermoplastic rubber (TPR)':'tpr','Poly Lactic Acid (PLA)': 'pla',\
    '100% Polyester':'polyester','100% UV Olefin':'olefin', '100% BCF Polypropylene': 'polypropylene',\
    '100% PVC':'pvc'}
        
    if s in replace_dict.keys():
        s=replace_dict[s]
    return s
    
tmp_material['value'] = tmp_material['value'].map(lambda x: change_material(x))

dict_materials = {}
key_list=tmp_material['product_uid'].keys()
for i in range(0,len(key_list)):
    if tmp_material['product_uid'][key_list[i]] not in dict_materials.keys():
        dict_materials[tmp_material['product_uid'][key_list[i]]]={}
        dict_materials[tmp_material['product_uid'][key_list[i]]]['product_uid']=tmp_material['product_uid'][key_list[i]]
        dict_materials[tmp_material['product_uid'][key_list[i]]]['cnt']=1
        dict_materials[tmp_material['product_uid'][key_list[i]]]['material']=tmp_material['value'][key_list[i]]
    else:
        ##print key_list[i]
        dict_materials[tmp_material['product_uid'][key_list[i]]]['material']=dict_materials[tmp_material['product_uid'][key_list[i]]]['material']+' '+tmp_material['value'][key_list[i]]
        dict_materials[tmp_material['product_uid'][key_list[i]]]['cnt']+=1
    if (i % 10000)==0:
        print i
                       
df_materials=pd.DataFrame(dict_materials).transpose()

### merge created 'material' column with df_all
df_all = pd.merge(df_all, df_materials[['product_uid','material']], how='left', on='product_uid')
df_all['material']=df_all['material'].fillna("").map(lambda x: x.encode('utf-8'))

df_all['material_parsed']=col_parser(df_all['material'].map(lambda x: x.replace("Other","").replace("*","")), parse_material=True,add_space_stop_list=[])

### list of all materials
list_materials=list(df_all['material_parsed'].map(lambda x: x.lower())) 



### count frequencies of materials in query and product_title
print "\nGenerating material dict: How many times each material appears in the dataset?"
material_dict=get_attribute_dict(list_materials,str_query=str_query)



### create dataframe and save to file
material_df=pd.DataFrame(material_dict).transpose()
material_df.to_csv(PROCESSINGTEXT_DIR+"/material_statistics_wo_google.csv")



### For further processing keep only materials that appear 
### more 10+ times in product_title and at least once in query
"""
for item in material_dict.keys():
    if (material_dict[item]['cnt_attribute']>=10 and material_dict[item]['cnt_query']>=1):
        1
    else:
        del(material_dict[item])
"""

for key in set(material_dict.keys()):
    if material_dict[key]['cnt_attribute']<20 or material_dict[key]['cnt_query']>3*material_dict[key]['cnt_attribute']:
        del(material_dict[key])



material_df=pd.DataFrame(material_dict).transpose().sort(['cnt_query'], ascending=[1])
print 'material dict creation time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

#################################################################
##### END OF COUNT MATERIALS ####################################
#################################################################


#################################################################
##### EXTRACT MATERIALS FROM QUERY AND PRODUCT TITLE ############
#################################################################

### At this moment we have parsed query and product title
### Now we will produce for query:
### brands_in_query, materials_in_query
### query_without_brand_names (we remove brand names from the text)
### query_without_brand_names_and_materials.
### Also, similar columns for product title.

def getremove_brand_or_material_from_str(s,df, replace_brand_dict={}):
    items_found=[]
    df=df.sort_values(['nwords'],ascending=[0])
    key_list=df['nwords'].keys()
    #start with several-word brands or materials
    #assert df['nwords'][key_list[0]]>1
    for i in range(0,len(key_list)):
        item=df['name'][key_list[i]]
        if item in s:
            if re.search(r'\b'+item+r'\b',s)!=None:
                s=re.sub(r'\b'+item+r'\b', '', s)
                if item in replace_brand_dict.keys():
                    items_found.append(replace_brand_dict[item])
                else:
                    items_found.append(item)

    return " ".join(s.split()), ";".join(items_found)


### We process only unique queries and product titles
### to reduce the processing time by more than 50%
aa=list(set(list(df_all['search_term_parsed'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=getremove_brand_or_material_from_str(aa[i],brand_df)
    if (i % 5000)==0:
        print "Extracted brands from",i,"out of",len(aa),"unique search terms; ", str(round((time()-t0)/60,1)),"minutes"
df_all['search_term_tuple']= df_all['search_term_parsed'].map(lambda x: my_dict[x])
df_all['search_term_parsed_woBrand']= df_all['search_term_tuple'].map(lambda x: x[0])
df_all['brands_in_search_term']= df_all['search_term_tuple'].map(lambda x: x[1])
print 'extract brands from query time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

df_all['search_term_tuple']= df_all['search_term_parsed_woBrand'].map(lambda x: getremove_brand_or_material_from_str(x,material_df))
df_all['search_term_parsed_woBM']= df_all['search_term_tuple'].map(lambda x: x[0])
df_all['materials_in_search_term']= df_all['search_term_tuple'].map(lambda x: x[1])
df_all=df_all.drop('search_term_tuple',axis=1)
print 'extract materials from query time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


##############################
aa=list(set(list(df_all['product_title_parsed'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=getremove_brand_or_material_from_str(aa[i],brand_df)
    if (i % 5000)==0:
        print "Extracted brands from",i,"out of",len(aa),"unique product titles; ", str(round((time()-t0)/60,1)),"minutes"

df_all['product_title_tuple']= df_all['product_title_parsed'].map(lambda x: my_dict[x])
df_all['product_title_parsed_woBrand']= df_all['product_title_tuple'].map(lambda x: x[0])
df_all['brands_in_product_title']= df_all['product_title_tuple'].map(lambda x: x[1])
print 'extract brands from product title time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


df_all['product_title_tuple']= df_all['product_title_parsed_woBrand'].map(lambda x: getremove_brand_or_material_from_str(x,material_df))
df_all['product_title_parsed_woBM']= df_all['product_title_tuple'].map(lambda x: x[0])
df_all['materials_in_product_title']= df_all['product_title_tuple'].map(lambda x: x[1])
df_all=df_all.drop('product_title_tuple',axis=1)
print 'extract materials from product titles time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


##### END OF EXTRACT MATERIALS FROM QUERY AND PRODUCT TITLE #####
#################################################################



###################################
##### Tagging #####################

### We use nltk.pos_tagger() to tag words
df_all['search_term_tokens'] =col_tagger(df_all['search_term_parsed_woBM'])
print 'search term tagging time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()
df_all['product_title_tokens'] =col_tagger(df_all['product_title_parsed_woBM'])
print 'product title tagging time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()



#################################################################
##### PROCESS ATTRIBUTES BULLETS ################################
#################################################################

### File attribute.csv contains 5343 different categories
### (https://www.kaggle.com/briantc/home-depot-product-search-relevance/homedepot-first-dataexploreation-k)
### Here we get we process only text in categories named 'Bullet##' where # stands for a number.
### This text is similar to product descriptions from 'product_descriptions.csv'.

### First, we concatenate all bullets for the same product_uid              
df_attr['product_uid']=df_attr['product_uid'].fillna(0)
df_attr['value']=df_attr['value'].fillna("")
df_attr['name']=df_attr['name'].fillna("")
dict_attr={}
for product_uid in list(set(list(df_attr['product_uid']))):
    dict_attr[int(product_uid)]={'product_uid':int(product_uid),'attribute_bullets':[]}

for i in range(0,len(df_attr['product_uid'])):
    if (i % 100000)==0:
        print "Read",i,"out of", len(df_attr['product_uid']), "rows in attributes.csv in", round((time()-t0)/60,1) ,'minutes'
    if df_attr['name'][i][0:6]=="Bullet":
        dict_attr[int(df_attr['product_uid'][i])]['attribute_bullets'].append(df_attr['value'][i])
            
if 0 in dict_attr.keys():
    del(dict_attr[0])
                        
for item in dict_attr.keys():
    if len(dict_attr[item]['attribute_bullets'])>0:
        dict_attr[item]['attribute_bullets']=". ".join(dict_attr[item]['attribute_bullets'])
        dict_attr[item]['attribute_bullets']+="."
    else:
        dict_attr[item]['attribute_bullets']=""

                                             
df_attr_bullets=pd.DataFrame(dict_attr).transpose()
df_attr_bullets['attribute_bullets']=df_attr_bullets['attribute_bullets'].map(lambda x: x.replace("..",".").encode('utf-8'))
print 'create attributes bullets time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()







### Then we follow similar steps as for query and product title above
### Parsing
df_attr_bullets['attribute_bullets_parsed'] = df_attr_bullets['attribute_bullets'].map(lambda x:str_parser(x,add_space_stop_list=[]))
print 'attribute bullets parsing time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

### Extracting brands...
df_attr_bullets['attribute_bullets_tuple']= df_attr_bullets['attribute_bullets_parsed'].map(lambda x: getremove_brand_or_material_from_str(x,brand_df))
df_attr_bullets['attribute_bullets_parsed_woBrand']= df_attr_bullets['attribute_bullets_tuple'].map(lambda x: x[0])
df_attr_bullets['brands_in_attribute_bullets']= df_attr_bullets['attribute_bullets_tuple'].map(lambda x: x[1])
print 'extract brands from attribute_bullets time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

### ... and materials from text...
df_attr_bullets['attribute_bullets_tuple']= df_attr_bullets['attribute_bullets_parsed_woBrand'].map(lambda x: getremove_brand_or_material_from_str(x,material_df))
df_attr_bullets['attribute_bullets_parsed_woBM']= df_attr_bullets['attribute_bullets_tuple'].map(lambda x: x[0])
df_attr_bullets['materials_in_attribute_bullets']= df_attr_bullets['attribute_bullets_tuple'].map(lambda x: x[1])
df_attr_bullets=df_attr_bullets.drop(['attribute_bullets_tuple'],axis=1)
print 'extract materials from attribute_bullets time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

### ... and tagging text using NLTK
df_attr_bullets['attribute_bullets_tokens'] =col_tagger(df_attr_bullets['attribute_bullets_parsed_woBM'])
print 'attribute bullets tagging time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

##### END OF PROCESS ATTRIBUTES BULLETS #########################
#################################################################

#################################################################
##### PROCESS PRODUCT DESCRIPTIONS ##############################
#################################################################

df_pro_desc = pd.read_csv(DATA_DIR+'/product_descriptions.csv')


### Parsing
df_pro_desc['product_description_parsed'] = df_pro_desc['product_description'].map(lambda x:str_parser(x,add_space_stop_list=add_space_stop_list).encode('utf-8'))
print 'product description parsing time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

### Extracting brands...
df_pro_desc['product_description_tuple']= df_pro_desc['product_description_parsed'].map(lambda x: getremove_brand_or_material_from_str(x,brand_df))
df_pro_desc['product_description_parsed_woBrand']= df_pro_desc['product_description_tuple'].map(lambda x: x[0])
df_pro_desc['brands_in_product_description']= df_pro_desc['product_description_tuple'].map(lambda x: x[1])
print 'extract brands from product_description time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

### ... and materials from text...
df_pro_desc['product_description_tuple']= df_pro_desc['product_description_parsed_woBrand'].map(lambda x: getremove_brand_or_material_from_str(x,material_df))
df_pro_desc['product_description_parsed_woBM']= df_pro_desc['product_description_tuple'].map(lambda x: x[0])
df_pro_desc['materials_in_product_description']= df_pro_desc['product_description_tuple'].map(lambda x: x[1])
df_pro_desc=df_pro_desc.drop(['product_description_tuple'],axis=1)
print 'extract materials from product_description time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

### ... and tagging text using NLTK
df_pro_desc['product_description_tokens'] = col_tagger(df_pro_desc['product_description_parsed_woBM'])
print 'product decription tagging time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

df_pro_desc['product_description']= df_pro_desc['product_description'].map(lambda x: x.encode('utf-8'))

#df_attr_bullets['attribute_bullets_stemmed']=df_attr_bullets['attribute_bullets_parsed'].map(lambda x:str_stemmer_wo_parser(x))
#df_attr_bullets['attribute_bullets_stemmed_woBM']=df_attr_bullets['attribute_bullets_parsed_woBM'].map(lambda x:str_stemmer_wo_parser(x))
#df_attr_bullets['attribute_bullets_stemmed_woBrand']=df_attr_bullets['attribute_bullets_parsed_woBrand'].map(lambda x:str_stemmer_wo_parser(x))
#df_pro_desc['product_description_stemmed']=df_pro_desc['product_description_parsed'].map(lambda x:str_stemmer_wo_parser(x))
#df_pro_desc['product_description_stemmed_woBM']=df_pro_desc['product_description_parsed_woBM'].map(lambda x:str_stemmer_wo_parser(x))
#df_pro_desc['product_description_stemmed_woBrand']=df_pro_desc['product_description_parsed_woBrand'].map(lambda x:str_stemmer_wo_parser(x))
#print 'stemming description and bullets time:',round((time()-t0)/60,1) ,'minutes\n'
#t0 = time()


##### END OF PROCESS PRODUCT DESCRIPTIONS #######################
#################################################################

#################################################################
##### GET IMPORTANT WORDS FROM QUERY AND PRODUCT TITLE ##########
#################################################################

### We started this work on our own by observing irregularities in models predictions,
### but we ended up with something similar to extracting the top trigram from
### http://blog.kaggle.com/2015/07/22/crowdflower-winners-interview-3rd-place-team-quartet/

### We found that some words are more important than the other 
### for predicting the relevance. For example, if the customer
### asks for 'kitchen faucet with side spray', she is looking for
### faucet, not for spray, side or kitchen. Therefore, faucets will 
### be more relevant for this query, but sprays, sides and kitchens
### will be less relevant.

### Let us define the most important word (or keyword) 'thekey'.
### The two words before it are 'beforethekey and 'before2thekey'.
### Example: query='outdoor ceiling fan with light'
### thekey='fan' 
### beforethekey='ceiling'
### before2thekey='outdoor'

### Below we build an algorithm to get such important words 
### from query and product titles.
### Our task is simplified due to (1) fairly uniform structure of 
### product titles and (2) small number of words in query.

### In the first step we delete irrelevant words using the following function.
### Although it may appear complex since we tried to correctly process as many
### entries as possible, but the basic logic is very simple:
### delete all words after 'with', 'for', 'in', 'that', 'on'
### as well as in some cases all words after colon ','
def cut_product_title(s):
    s=s.lower()
    s = re.sub('&amp;', '&', s)
    s = re.sub('&nbsp;', '', s)
    s = re.sub('&#39;', '', s)
    s = re.sub(r'(?<=[0-9]),[\ ]*(?=[0-9])', '', s)
    s = re.sub(r'(?<=\))(?=[a-zA-Z0-9])', ' ', s) # add space between parentheses and letters
    s = re.sub(r'(?<=[a-zA-Z0-9])(?=\()', ' ', s) # add space between parentheses and letters
    s = s.replace(";",". ")
    s = s.replace(":"," . ")
    s=s.replace("&"," and ") 
    s = re.sub('[^a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,\+]', ' ', s)
    s= " ".join(s.split())
    s = re.sub(r'(?<=[0-9])\.\ ', ' ', s)
    s = re.sub(r'(?<=\ in)\.(?=[a-zA-Z])', '. ', s)
    
    s=replace_in_parser(s)
    
    s = re.sub(r'\-discontinued', '', s)
    s = re.sub(r' \+ free app(?=$)', '', s)
    s = s.replace("+"," ")    
    s = re.sub('\([a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]+?\)', '', s)
    #s= re.sub('[\(\)]', '', s)
    if " - " in s:
        #srch=re.search(r'(?<= - )[a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]+',s) 
        if re.search(r'(\d|\.|mm|cm|in|ft|mhz|volt|\bhp|\bl|oz|lb|gal) \- \d',s)==None \
            and re.search(r' (sign|carpet|decal[s]*|figure[s]*)(?=$)',s)==None and re.search(r'\d \- (way\b|day\b)',s)==None: 
        #if ' - ' is found and the string doesnt end with word 'sign' or 'carpet' or 'decal' and not string '[0-9] - way' found
            s = re.sub(r' - [a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]*', '', s) #greedy regular expression
    if "uilt in" not in s and "uilt In" not in s:
        s = re.sub(r'(?<=[a-zA-Z\%\$\#\@\&\/\.\*])[\ ]+[I|i]n [a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]*', '', s)
    s = s.replace(" - "," ")
    if re.search(r' (sign|decal[s]*|figure[s]*)(?=$)',s)==None:    
        s = re.sub(r'(?<=[a-zA-Z0-9\%\$\#\@\&\/\.\*])[\ ]+[W|w]ith [a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]*', '', s)
        s = re.sub(r'(?<=[a-zA-Z0-9\%\$\#\@\&\/\.\*])[\ ]+[W|w]ithout [a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]*', '', s)
    s = re.sub(r'(?<=[a-zA-Z\%\$\#\@\&\/\.\*])[\ ]+[w]/[\ a-z0-9][a-z0-9][a-z0-9\.][a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]*', '', s)
    
    if " fits for " not in s and " for fits " not in s:    
        s = re.sub(r'(?<=[a-zA-Z0-9\%\$\#\@\&\/\.\*])[\ ]+fits [a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]*', '', s)    
    if " for lease " not in s and re.search(r' (sign|decal[s]*|figure[s]*)(?=$)',s)==None:
        s = re.sub(r'(?<=[a-zA-Z0-9\%\$\#\@\&\/\.\*])[\ ]+[F|f]or [a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]*', '', s)
    s = re.sub(r'(?<=[a-zA-Z0-9\%\$\#\@\&\/\.\*])[\ ]+[T|t]hat [a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]*', '', s)


    s = re.sub(r' on (wheels|a pallet|spool|bracket|3 in|blue post|360|track|spike|rock|lamp|11 in|2 in|pedestal|square base|tub|steel work)[a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)]*', '', s)
    s = re.sub(r' on (plinth|insulator|casters|pier base|reel|fireplace|moon|bracket|24p ebk|zinc spike|mailbox|cream chand|blue post)[a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)]*', '', s)
 
    s = re.sub(r'(?<= on white stiker)[a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]*', '', s)
    s = re.sub(r' on [installing][a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]*', '', s)
    if "," in s:
        srch=re.search(r'(?<=, )[a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]*',s)  #greedy regular expression
        if srch!=None:
            if len(re.sub('[^a-zA-Z\ ]', '', srch.group(0)))<25:
                s = re.sub(r', [a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]*', '', s)
    s = re.sub(r'(?<=recessed door reinforcer), [a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]*', '', s)
    #s = re.sub(r'(?<=[a-zA-Z0-9]),\ [a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)]*', '', s)
    s = re.sub(r'(?<=[a-zA-Z\%\$\#\@\&\/\.\*]) [F|f]eaturing [a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)]*', '', s)
    s = re.sub(r'(?<=[a-zA-Z\%\$\#\@\&\/\.\*]) [I|i]ncludes [a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)]*', '', s)
    s = re.sub(' [\#]\d+[\-\d]*[\,]*', '', s)  
    s = re.sub(r'(?<=[a-zA-Z\ ])\/(?=[a-zA-Z])', ' ', s)
    s = re.sub(r'(?<=[a-zA-Z\ ])\-(?=[a-zA-Z])', ' ', s)
    s = s.replace(",",". ")
    s = s.replace("..",".")
    s = s.replace("..",".")
    s = s.replace("*","")
    return " ".join([word.replace("-","") for word in s.split() if re.search(r'\d\-\d',word)==None])
    
    
### The next step is identify the most important words. 
### We exclude brand names and similar words like 'EpoxyShield' 
### (see how add_space_stop_list is created)
not_keyword_list=list(brand_df['name'][brand_df['nwords']==1])
for item in add_space_stop_list:
    if len(wn.synsets(item,pos=wn.NOUN))==0:
        not_keyword_list.append(item)

### We want 'thekey' to be a noun as identified by NLTK WordNet
### and NN, NNS, VBG in the sentence as identified by NLTK.pos_tagger()
### Since pos_tagger often fails, we run it two times: on full sentence with
### punctuation and on separate words. We reject the word only if in neither run
### it is identified as NN, NNS, VBG.
### We  also have to create in_list 
### with words that are always to be identified as thekeys. Words ending with 
### '-er', '-ers', '-or', '-ors' are also thekeys. 
### We exclude some words from potential thekeys, they are added to out_list.

### Once thekey is identified, we read the words to the left and consider
### them as keywords (or important words). If we encounter nouns, we continue.
### IF we encounter ['JJ','JJS', 'JJR', 'RB', 'RBS', 'RBR', 'VBG', 'VBD', 'VBN','VBP'],
### we add the word to the keywords, but stop unless the next word is 'and'.
### In other cases (word with digits or preposition etc) we just stop.

# the word lists from the following function are shared on the forum.
def get_key_words(tag_list,wordtag_list, string_output=False,out_list=not_keyword_list[:]):
    i=len(tag_list)
    in_list=['tv','downrod', 'sillcock', 'shelving', 'luminaire', 'paracord', 'ducting', \
    'recyclamat', 'rebar', 'spackling', 'hoodie', 'placemat', 'innoculant', 'protectant', \
    'colorant', 'penetrant', 'attractant', 'bibb', 'nosing', 'subflooring', 'torchiere', 'thhn',\
    'lantern','epoxy','cloth','trim','adhesive','light','lights','saw','pad','polish','nose','stove',\
    'elbow','elbows','lamp','door','doors','pipe','bulb','wood','woods','wire','sink','hose','tile','bath','table','duct',\
    'windows','mesh','rug','rugs','shower','showers','wheels','fan','lock','rod','mirror','cabinet','shelves','paint',\
    'plier','pliers','set','screw','lever','bathtub','vacuum','nut', 'nipple','straw','saddle','pouch','underlayment',\
    'shade','top', 'bulb', 'bulbs', 'paint', 'oven', 'ranges', 'sharpie', 'shed', 'faucet',\
    'finish','microwave', 'can', 'nozzle', 'grabber', 'tub', 'angles','showerhead', 'dehumidifier', 'shelving', 'urinal', 'mdf']

    out_list= out_list +['free','height', 'width', 'depth', 'model','pcs', 'thick','pack','adhesive','steel','cordless', 'aaa' 'b', 'nm', 'hc', 'insulated','gll', 'nutmeg',\
    'pnl', 'sotc','withe','stainless','chrome','beige','max','acrylic', 'cognac', 'cherry', 'ivory','electric','fluorescent', 'recessed', 'matte',\
    'propane','sku','brushless','quartz','gfci','shut','sds','value','brown','white','black','red','green','yellow','blue','silver','pink',\
    'gray','gold','thw','medium','type','flush',"metaliks", 'metallic', 'amp','btu','gpf','pvc','mil','gcfi','plastic', 'vinyl','aaa',\
    'aluminum','brass','antique', 'brass','copper','nickel','satin','rubber','porcelain','hickory','marble','polyacrylic','golden','fiberglass',\
    'nylon','lmapy','maple','polyurethane','mahogany','enamel', 'enameled', 'linen','redwood', 'sku','oak','quart','abs','travertine', 'resin',\
    'birch','birchwood','zinc','pointe','polycarbonate', 'ash', 'wool', 'rockwool', 'teak','alder','frp','cellulose','abz', 'male', 'female', 'used',\
    'hepa','acc','keyless','aqg','arabesque','polyurethane', 'polyurethanes','ardex','armorguard','asb', 'motion','adorne','fatpack',\
    'fatmax','feet','ffgf','fgryblkg', 'douglas', 'fir', 'fleece','abba', 'nutri', 'thermal','thermoclear', 'heat', 'water', 'systemic',\
    'heatgasget', 'cool', 'fusion', 'awg', 'par', 'parabolic', 'tpi', 'pint', 'draining', 'rain', 'cost', 'costs', 'costa','ecostorage',
    'mtd', 'pass', 'emt', 'jeld', 'npt', 'sch', 'pvc', 'dusk', 'dawn', 'lathe','lows','pressure', 'round', 'series','impact', 'resistant','outdoor',\
    'off', 'sawall', 'elephant', 'ear', 'abb', 'baby', 'feedback', 'fastback','jumbo', 'flexlock', 'instant', 'natol', 'naples','florcant',\
    'canna','hammock', 'jrc', 'honeysuckle', 'honey', 'serrano','sequoia', 'amass', 'ashford', 'gal','gas', 'gasoline', 'compane','occupancy',\
    'home','bakeware', 'lite', 'lithium', 'golith','gxwh',  'wht', 'heirloom', 'marine', 'marietta', 'cambria', 'campane','birmingham',\
    'bellingham','chamois', 'chamomile', 'chaosaw', 'chanpayne', 'thats', 'urethane', 'champion', 'chann', 'mocha', 'bay', 'rough',\
    'undermount', 'price', 'prices', 'way', 'air', 'bazaar', 'broadway', 'driveway', 'sprayway', 'subway', 'flood', 'slate', 'wet',\
    'clean', 'tweed', 'weed', 'cub', 'barb', 'salem', 'sale', 'sales', 'slip', 'slim', 'gang', 'office', 'allure', 'bronze', 'banbury',\
    'tuscan','tuscany', 'refinishing', 'fleam','schedule', 'doeskin','destiny', 'mean', 'hide', 'bobbex', 'pdi', 'dpdt', 'tri', 'order',\
    'kamado','seahawks','weymouth', 'summit','tel','riddex', 'alick','alvin', 'ano', 'assy', 'grade', 'barranco', 'batte','banbury',\
    'mcmaster', 'carr', 'ccl', 'china', 'choc', 'colle', 'cothom', 'cucbi', 'cuv', 'cwg', 'cylander', 'cylinoid', 'dcf', 'number', 'ultra',\
    'diat','discon', 'disconnect', 'plantation', 'dpt', 'duomo', 'dupioni', 'eglimgton', 'egnighter','ert','euroloft', 'everready',\
    'felxfx', 'financing', 'fitt', 'fosle', 'footage', 'gpf','fro', 'genis', 'giga', 'glu', 'gpxtpnrf', 'size', 'hacr', 'hardw',\
    'hexagon', 'hire', 'hoo','number','cosm', 'kelston', 'kind', 'all', 'semi', 'gloss', 'lmi', 'luana', 'gdak', 'natol', 'oatu',\
    'oval', 'olinol', 'pdi','penticlea', 'portalino', 'racc', 'rads', 'renat', 'roc', 'lon', 'sendero', 'adora', 'sleave', 'swu',
    'tilde', 'cordoba', 'tuvpl','yel', 'acacia','mig','parties','alkaline','plexiglass', 'iii', 'watt']
    
    output_list=[]
    if i>0:
        finish=False
        started = False
        while not finish:
            i-=1
            
            if started==False:
                if (wordtag_list[i][0] not in out_list) \
                and (wordtag_list[i][0] in in_list \
                    or (re.search(r'(?=[e|o]r[s]*\b)',wordtag_list[i][0])!=None and re.search(r'\d+',wordtag_list[i][0])==None) \
                    or (len(wordtag_list[i][0])>2 and re.search(r'\d+',wordtag_list[i][0])==None and len(wn.synsets(wordtag_list[i][0],pos=wn.NOUN))>0 \
                        and (wordtag_list[i][1] in ['NN', 'NNS','VBG'] or tag_list[i][1] in ['NN', 'NNS','VBG']) \
                            and len(re.sub('[^aeiouy]', '', wordtag_list[i][0]))>0 )): #exclude VBD
                    started = True
                    output_list.insert(0,wordtag_list[i])
                        # handle exceptions below
                        # 'iron' only with -ing is OK: soldering iron, seaming iron
                    if i>1 and wordtag_list[i][0] in ['iron','irons'] and re.search(r'ing\b',wordtag_list[i-1][0])==None:
                        output_list=[]
                        started = False    
            else:

                if tag_list[i][1] in ['NN','NNP', 'NNPS', 'NNS']:
                    if len(re.sub('[^0-9]', '', tag_list[i][0]))==0 and \
                    len(re.sub('[^a-zA-Z0-9\-]', '', tag_list[i][0]))>2 \
                    and tag_list[i][0] not in ['amp','btu','gpf','pvc','mil','watt','gcfi']\
                    and (len(wn.synsets(tag_list[i][0]))>0 or re.search(r'(?=[e|o]r[s]*\b)',tag_list[i][0])!=None):
                        output_list.insert(0,tag_list[i])
                elif tag_list[i][0]=='and':
                    output_list.insert(0,tag_list[i])
                    started=False
                else:
                    if tag_list[max(0,i-1)][0]!="and" and (tag_list[i][1] not in ['VBD', 'VBN']):
                        finish=True
                    if tag_list[i][1] in ['JJ','JJS', 'JJR', 'RB', 'RBS', 'RBR', 'VBG', 'VBD', 'VBN','VBP']:
                        if len(re.sub('[^0-9]', '', tag_list[i][0]))==0 and \
                        len(re.sub('[^a-zA-Z0-9\-]', '', tag_list[i][0]))>2 \
                        and tag_list[i][0] not in ['amp','btu','gpf','pvc','mil','watt','gcfi']\
                        and len(wn.synsets(tag_list[i][0]))>0:
                            output_list.insert(0,tag_list[i])
                
            if i==0:
                finish=True
    if string_output==True:
        return " ".join([tag[0] for tag in output_list])
    else:
        return output_list



### Apply the function to product_title
### We have to start with product_title, not product_title_parsed,
### since punctuation is important for our task ...
df_all['product_title_cut']= df_all['product_title'].map(lambda x: cut_product_title(x).encode('utf-8'))

### ... and that is why we have to remove the brand names again
aa=list(set(list(df_all['product_title_cut'])))
my_dict={}
for i in range(0,len(aa)):
    my_dict[aa[i]]=getremove_brand_or_material_from_str(aa[i],brand_df)
    if (i % 5000)==0:
        print "processed "+str(i)+" out of "+str(len(aa))+" unique cut product titles; "+str(round((time()-t0)/60,1))+" minutes"
df_all['product_title_cut_tuple']= df_all['product_title_cut'].map(lambda x: my_dict[x])
df_all['product_title_cut_woBrand']= df_all['product_title_cut_tuple'].map(lambda x: x[0])
df_all=df_all.drop(['product_title_cut_tuple'],axis=1)
print 'extract brands from cut product title:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

### Tagging two times: full sentences and separate words
df_all['product_title_cut_tokens'] =col_tagger(df_all['product_title_cut_woBrand'])
df_all['product_title_cut_wordtokens'] =col_wordtagger(df_all['product_title_cut_woBrand'])

### the same steps for search term, but we now we continue with the preprocessed resuts
### since punctuation is not as important in query as it is in product title
df_all['search_term_cut_woBrand']= df_all['search_term_parsed_woBrand'].map(lambda x: cut_product_title(x).encode('utf-8'))
df_all['search_term_cut_tokens'] =col_tagger(df_all['search_term_cut_woBrand'])
df_all['search_term_cut_wordtokens'] =col_wordtagger(df_all['search_term_cut_woBrand'])

### Transform tags into text, it may look like unecessary step.
### But in our work we have to frequently save processing results and recover tags from text.
### Here this transformation is used to make the _tokens variables compatibe with 
### parser_mystr2tuple() function 
df_all['search_term_cut_tokens']=df_all['search_term_cut_tokens'].map(lambda x: str(x))
df_all['search_term_cut_wordtokens']=df_all['search_term_cut_wordtokens'].map(lambda x: str(x))
df_all['product_title_cut_tokens']=df_all['product_title_cut_tokens'].map(lambda x: str(x))
df_all['product_title_cut_wordtokens']=df_all['product_title_cut_wordtokens'].map(lambda x: str(x))


df_all['search_term_keys']=df_all.apply(lambda x: \
            get_key_words(parser_mystr2tuple(x['search_term_cut_tokens']),parser_mystr2tuple(x['search_term_cut_wordtokens']),string_output=True),axis=1)
df_all['product_title_keys']=df_all.apply(lambda x: \
            get_key_words(parser_mystr2tuple(x['product_title_cut_tokens']),parser_mystr2tuple(x['product_title_cut_wordtokens']),string_output=True),axis=1)



### Now we just need to assing the last word from keywords as thekey,
### the words before it as beforethekey and before2thekey.
### One more trick: we first get this trigram from product_title,
### than use thekey from product title to choose the most similar word
### in case query contains two candidates separated by 'and'
### For example, for query 'microwave and stove' we may chose either 
### 'microwave' or 'stove' depending on the thekey from product title.

def get_last_words_from_parsed_title(s):
    words=s.split()
    if len(words)==0:
        last_word=""
        word_before_last=""
        word2_before_last=""
    else:
        last_word=words[len(words)-1]
        word_before_last=""
        word2_before_last=""
        if len(words)>1:
            word_before_last=words[len(words)-2]
            if word_before_last=="and":
                word_before_last=""
            if len(words)>2 and word_before_last!="and":
                word2_before_last=words[len(words)-3]
                if word2_before_last=="and":
                    word2_before_last=""
    return last_word, word_before_last, word2_before_last

def get_last_words_from_parsed_query(s,last_word_in_title):
    words=s.split()
    if len(words)==0:
        last_word=""
        word_before_last=""
        word2_before_last=""
    else:
        last_word=words[len(words)-1]
        word_before_last=""
        word2_before_last=""
        if len(words)>1:
            word_before_last=words[len(words)-2]
            
            if len(words)>2 and word_before_last!="and":
                word2_before_last=words[len(words)-3]
                if word2_before_last=="and":
                    word2_before_last=""
                    
            if word_before_last=="and":
                word_before_last=""
                if len(words)>2:
                    cmp_word=words[len(words)-3]
                    sm1=find_similarity(last_word,last_word_in_title)[0]
                    sm2=find_similarity(cmp_word,last_word_in_title)[0]
                    if sm1<sm2:
                        last_word=cmp_word
                        if len(words)>3:
                            word_before_last=words[len(words)-4]
               
            
    return last_word, word_before_last, word2_before_last



### get trigram from product title
df_all['product_title_thekey_tuple']=df_all['product_title_keys'].map(lambda x: get_last_words_from_parsed_title(x))
df_all['product_title_thekey']=df_all['product_title_thekey_tuple'].map(lambda x: x[0])
df_all['product_title_beforethekey']=df_all['product_title_thekey_tuple'].map(lambda x: x[1])
df_all['product_title_before2thekey']=df_all['product_title_thekey_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['product_title_thekey_tuple'],axis=1)


### get trigram from query
df_all['search_term_thekey_tuple']=df_all.apply(lambda x: \
            get_last_words_from_parsed_query(x['search_term_keys'],x['product_title_thekey']),axis=1)
#df_all['thekey_info']=df_all['search_term_keys']+"\t"+df_all['product_title_thekey']
#df_all['search_term_thekey_tuple']=df_all['thekey_info'].map(lambda x: get_last_words_from_parsed_query(x.split("\t")[0],x.split("\t")[1]))
df_all['search_term_thekey']=df_all['search_term_thekey_tuple'].map(lambda x: x[0])
df_all['search_term_beforethekey']=df_all['search_term_thekey_tuple'].map(lambda x: x[1])
df_all['search_term_before2thekey']=df_all['search_term_thekey_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['search_term_thekey_tuple'],axis=1)

#df_all['search_term_thekey_stemmed']=df_all['search_term_thekey'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
#df_all['product_title_thekey_stemmed']=df_all['product_title_thekey'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
#df_all['search_term_beforethekey_stemmed']=df_all['search_term_beforethekey'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
#df_all['product_title_beforethekey_stemmed']=df_all['product_title_beforethekey'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
#df_all['search_term_before2thekey_stemmed']=df_all['search_term_before2thekey'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
#df_all['product_title_before2thekey_stemmed']=df_all['product_title_before2thekey'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))

print 'extracting important words time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


##### END OF GET IMPORTANT WORDS FROM QUERY AND PRODUCT TITLE ###
#################################################################

#################################################################
##### STEMMING ##################################################
#################################################################

### We also exclude stopwords here.
### Sometimes people search for 'can' with the meaning 'a container'
### like in 'trash can'. That is why we keep 'can' in query and product title.
df_attr_bullets['attribute_bullets_stemmed']=df_attr_bullets['attribute_bullets_parsed'].map(lambda x:str_stemmer_wo_parser(x))
df_attr_bullets['attribute_bullets_stemmed_woBM']=df_attr_bullets['attribute_bullets_parsed_woBM'].map(lambda x:str_stemmer_wo_parser(x))
df_attr_bullets['attribute_bullets_stemmed_woBrand']=df_attr_bullets['attribute_bullets_parsed_woBrand'].map(lambda x:str_stemmer_wo_parser(x))
df_pro_desc['product_description_stemmed']=df_pro_desc['product_description_parsed'].map(lambda x:str_stemmer_wo_parser(x))
df_pro_desc['product_description_stemmed_woBM']=df_pro_desc['product_description_parsed_woBM'].map(lambda x:str_stemmer_wo_parser(x))
df_pro_desc['product_description_stemmed_woBrand']=df_pro_desc['product_description_parsed_woBrand'].map(lambda x:str_stemmer_wo_parser(x))
df_all['search_term_keys_stemmed']=df_all['search_term_keys'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
df_all['product_title_keys_stemmed']=df_all['product_title_keys'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
df_all['search_term_stemmed']=df_all['search_term_parsed'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
df_all['search_term_stemmed_woBM']=df_all['search_term_parsed_woBM'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
df_all['search_term_stemmed_woBrand']=df_all['search_term_parsed_woBrand'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
df_all['product_title_stemmed']=df_all['product_title_parsed'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
df_all['product_title_stemmed_woBM']=df_all['product_title_parsed_woBM'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
df_all['product_title_stemmed_woBrand']=df_all['product_title_parsed_woBrand'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
df_all['search_term_thekey_stemmed']=df_all['search_term_thekey'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
df_all['product_title_thekey_stemmed']=df_all['product_title_thekey'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
df_all['search_term_beforethekey_stemmed']=df_all['search_term_beforethekey'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
df_all['product_title_beforethekey_stemmed']=df_all['product_title_beforethekey'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
df_all['search_term_before2thekey_stemmed']=df_all['search_term_before2thekey'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))
df_all['product_title_before2thekey_stemmed']=df_all['product_title_before2thekey'].map(lambda x: str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))

print 'stemming time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


##### END OF STEMMING ###########################################
#################################################################


### Save everything into files
df_all['product_title']= df_all['product_title'].map(lambda x: x.encode('utf-8'))
df_all.to_csv(PROCESSINGTEXT_DIR+"/df_train_and_test_processed_wo_google.csv", index=False)

df_attr_bullets.to_csv(PROCESSINGTEXT_DIR+"/df_attribute_bullets_processed_wo_google.csv", index=False)
df_pro_desc.to_csv(PROCESSINGTEXT_DIR+"/df_product_descriptions_processed_wo_google.csv", index=False)


print 'TOTAL PROCESSING TIME:',round((time()-t1)/60,1) ,'minutes\n'
t1 = time()

df_all=df_all.drop(list(df_all.keys()),axis=1)
df_attr_bullets=df_attr_bullets.drop(list(df_attr_bullets.keys()),axis=1)
df_pro_desc=df_pro_desc.drop(list(df_pro_desc.keys()),axis=1)







