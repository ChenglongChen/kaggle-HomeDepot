# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: generate all the data and features in one shot
@note: if you don't have access to multi-core computers, drop the "&" in the cmd

"""

import os


#-----------------------------------------------------------------------
# generate split (or you can use mine in ./Data/split/)
cmd = "python splitter.py"
os.system(cmd)


#-----------------------------------------------------------------------
# prepare data
cmd = "python data_preparer.py"
os.system(cmd)


#-----------------------------------------------------------------------
# process/clean data
cmd = "python data_processor.py"
os.system(cmd)


#-----------------------------------------------------------------------
# generate basic features
cmd = "python feature_basic.py &"
os.system(cmd)


#-----------------------------------------------------------------------
# generate distance features
cmd = "python feature_distance.py jaccard &"
os.system(cmd)

cmd = "python feature_distance.py edit &"
os.system(cmd)

# # not used in the final model
# cmd = "python feature_distance.py compression &"
# os.system(cmd)


#-----------------------------------------------------------------------
# generate first and last ngram features
cmd = "python feature_first_last_ngram.py &"
os.system(cmd)


#-----------------------------------------------------------------------
# generate group based features (not used in the final model)
# cmd = "python feature_group_distance.py &"
# os.system(cmd)

# cmd = "python feature_group_distance_stat.py &"
# os.system(cmd)

cmd = "python feature_group_relevance.py &"
os.system(cmd)


#-----------------------------------------------------------------------
# generate intersect features
cmd = "python feature_intersect_count.py &"
os.system(cmd)

cmd = "python feature_intersect_position.py &"
os.system(cmd)


#-----------------------------------------------------------------------
# generate match features
cmd = "python feature_match.py &"
os.system(cmd)


#-----------------------------------------------------------------------
# generate query quality features
cmd = "python feature_query_quality.py &"
os.system(cmd)


#-----------------------------------------------------------------------
# generate statistical cooccurrence (weighted) features
cmd = "python feature_stat_cooc_tfidf.py tf &"
os.system(cmd)
# cmd = "python feature_stat_cooc_tfidf.py norm_tf &"
# os.system(cmd)

cmd = "python feature_stat_cooc_tfidf.py tfidf &"
os.system(cmd)
# cmd = "python feature_stat_cooc_tfidf.py norm_tfidf &"
# os.system(cmd)

cmd = "python feature_stat_cooc_tfidf.py bm25 &"
os.system(cmd)


#-----------------------------------------------------------------------
# generate word2vec features using pre-trained word2vec model
cmd = "python feature_word2vec.py google &"
os.system(cmd)

cmd = "python feature_word2vec.py wikipedia &"
os.system(cmd)


#-----------------------------------------------------------------------
# generate wordnet similarity features
# time consuming part ~20 hrs
cmd = "python feature_wordnet_similarity.py &"
os.system(cmd)


#-----------------------------------------------------------------------
# generate word2vec & doc2vec features using word2vec/doc2vec models
# trained with Homedepot provided data
cmd = "python embedding_trainer.py"
os.system(cmd)

# most time consuming part 1 ~ 2 days
# after you have trained the WORD2VEC model above, you can MANUALLY distribute the
# feature generation part in feature_word2vec.py to save time, e.g.,
# you can run the following part in parallel
# - search_term vs product_title
# - search_term vs product_description
# - search_term vs product_attribute
cmd = "python feature_word2vec.py homedepot &"
os.system(cmd)

cmd = "python feature_doc2vec.py &"
os.system(cmd)


#-----------------------------------------------------------------------
# generate vector space features
# most memory consuming part > 16GB
cmd = "python feature_vector_space.py"
os.system(cmd)

cmd = "python convert_pkl_lsa_to_csv_lsa.py"
os.system(cmd)

cmd = "Rscript feature_tsne.R"
os.system(cmd)

cmd = "python convert_csv_tsne_to_pkl_tsne.py"
os.system(cmd)
