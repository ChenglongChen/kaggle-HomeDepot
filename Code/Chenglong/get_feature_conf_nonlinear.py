# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: generate feature conf for the following models
        - reg_skl_gbm
        - reg_skl_adaboost
        - reg_skl_etr
        - reg_skl_rf
        - reg_xgb_tree
        - reg_rgf
@note: 
        - such features INCLUDE "DocId_(search_term|product_title|product_color|product_brand)"
        - one can tune the MANDATORY_FEATS and COMMENT_OUT_FEATS to generate different feature subset

"""

import re
import os
from optparse import OptionParser

import config
from utils import time_utils


INCLUDE_FEATS = [
".+"
]


COUNT_FEATS = [
"Freq", 
"Len", 
"Count", 
"Size", 
"Position", 
]
# COUNT_FEATS = []


NOT_COUNT_FEATS = ["Norm", "Ratio"]


MANDATORY_FEATS = [

"DocId_(search_term|product_title|product_color|product_brand)",

# including product_uid according to
# https://www.kaggle.com/c/home-depot-product-search-relevance/forums/t/20288/trends-in-relevances-by-row-ids/115886#post115886
"DocIdEcho_product_uid",
"ProductUidDummy1_product_uid",
"ProductUidDummy2_product_uid",

"IsInGoogleDict",
"GroupRelevance_Size",
"TSNE",
]


COMMENT_OUT_FEATS = [

#-------------- General --------------
"search_term_alt",

"Bigram",
"Trigram",
"UBgram",
"UBTgram",

"Median",
"Std",

".+(Bigram|Trigram)_.+_product_(brand|color)",


#-------------- Basic --------------
"DocLogFreq",
"Digit",
"Unique",
"^DocIdOneHot",
"^DocId",

"DocLen_product_(brand|color)",
"DocLen_product_attribute_1D",
"DocFreq_product_description_1D",
"DocFreq_product_attribute_1D",
"Digit(Count|Ratio)_product_(brand|color)",
"Doc(Entropy|Len)_product_(brand|color)",
"Unique(Count|Ratio)_.+_product_(brand|color)",


#-------------- Distance --------------
"DiceDistance",
# "EditDistance",
"Compression",


#-------------- First and Last Ngram --------------
"FirstIntersectNormPosition",
"FirstIntersectPosition",
"LastIntersectNormPosition",
"LastIntersectPosition",


#-------------- Group --------------
"GroupRelevance_(Mean|Std|Max|Min|Median)",
"Group_\d+",
"GroupDistanceStat",


#-------------- Intersect Count & Position --------------
"IntersectPosition_.+_(Std|Max|Min|Median)",
"IntersectNormPosition_.+_(Std|Max|Min|Median)",


#-------------- Match --------------
"LongestMatchSize",


#-------------- StatCooc --------------
# since product_name is of length 2, it makes no difference for various aggregation as there is only one item
"StatCooc(TF|NormTF|TFIDF|NormTFIDF|BM25)_Bigram_(Std|Max|Min|Median)_search_term_product_name_x_product_title_product_name_1D",
"StatCooc(TF|NormTF|TFIDF|NormTFIDF|BM25)_Bigram_(Std|Max|Min|Median)_product_title_product_name_x_search_term_product_name_1D",

"NormTF",
"NormTFIDF",


#-------------- Vector Space --------------
# as TFIDF_Word_Trigram has the largest corr
"LSA\d+_Word_Unigram",
"LSA\d+_Word_Bigram",
"TFIDF_Word_Unigram",
"TFIDF_Word_Bigram",

# as TFIDF_Char_Fourgram has the largest corr
"LSA\d+_Char_Bigram",
"LSA\d+_Char_Trigram",
"LSA\d+_Char_Fivegram",
"TFIDF_Char_Bigram",
"TFIDF_Char_Trigram",
"TFIDF_Char_Fivegram",

"CharDistribution_Ratio",


#-------------- Word2Vec & Doc2Vec --------------
"_Vector_", 
"_Vdiff_", 
"Word2Vec_Wikipedia_D50",
"Word2Vec_Wikipedia_D100",
"Word2Vec_Wikipedia_D200",
# "Word2Vec_GoogleNews",
"Word2Vec_GoogleNews_D300_Vector",
# as all the words are used to train the model
"Word2Vec_Homedepot_D100_Importance",
"Word2Vec_Homedepot_D100_N_Similarity_Imp",


#-------------- Turing Test --------------
# d = {
#     "df_basic_features.csv": "Basic",
#     "df_brand_material_dummies.csv": "BrandMaterialDummy",
#     "df_dist_new.csv": "Dist",
#     "dld_features.csv": "DLD",
#     "df_st_tfidf.csv": "StTFIDF",
#     "df_tfidf_intersept_new.csv": "TFIDF",
#     "df_thekey_dummies.csv": "TheKeyDummy",
#     "df_word2vec_new.csv": "Word2Vec",
# }
# "TuringTest_Basic",
# "TuringTest_BrandMaterialDummy",
# "TuringTest_Dist",
# "TuringTest_DLD",
# "TuringTest_StTFIDF",
# "TuringTest_TFIDF",
# "TuringTest_TheKeyDummy",
# "TuringTest_Word2Vec",


]


def _check_include(fname):
    for v in INCLUDE_FEATS:
        pat = re.compile(v)
        if len(re.findall(pat, fname)) > 0:
            return True
    return False


def _check_count_feat(fname):
    for v in NOT_COUNT_FEATS:
        pat = re.compile(v)
        if len(re.findall(pat, fname)) > 0:
            return False
    for v in COUNT_FEATS:
        pat = re.compile(v)
        if len(re.findall(pat, fname)) > 0:
            return True
    return False


def _check_lsa_matrix(fname):
    pat = re.compile("^LSA")
    if len(re.findall(pat, fname)) > 0:
        return True
    return False


def _check_mandatory(fname):
    for v in MANDATORY_FEATS:
        pat = re.compile(v)
        if len(re.findall(pat, fname)) > 0:
            return True
    return False


def _check_comment_out(fname):
    for v in COMMENT_OUT_FEATS:
        pat = re.compile(v)
        if len(re.findall(pat, fname)) > 0:
            return True
    return False


header_pattern = """
# -*- coding: utf-8 -*-
\"\"\"
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: one feature conf

Generated by
python %s -d %d -o %s

Format:
FEATURE_NAME : (MANDATORY, TRANSFORM)

\"\"\"

import config
from feature_transformer import SimpleTransform, ColumnSelector

LSA_COLUMNS = range(%d)

feature_dict = {

"""


def _create_feature_conf(lsa_columns, outfile):
    res = header_pattern%(__file__, int(lsa_columns), outfile, int(lsa_columns))

    folders = [config.FEAT_DIR, config.FEAT_DIR+"/All"]
    for folder in folders:
        try:
            for file in sorted(os.listdir(folder)):
                if config.FEAT_FILE_SUFFIX in file:
                    fname = file.split(".")[0]
                    if _check_include(fname):
                        line = ""
                        mandatory = _check_mandatory(fname)
                        if not mandatory and _check_comment_out(fname):
                            continue
                            line += "# "
                        line += "'%s' : "%fname
                        if mandatory:
                            line += "(True, "
                        else:
                            line += "(False, "
                        if _check_lsa_matrix(fname):
                            if int(lsa_columns) > 0:
                                line += "ColumnSelector(LSA_COLUMNS)),\n"
                            else:
                                continue
                        elif _check_count_feat(fname):
                            line += "SimpleTransform(config.COUNT_TRANSFORM)),\n"
                        else:
                            line += "SimpleTransform()),\n"
                        res += line
        except:
            pass
    res += "}\n"

    with open(os.path.join(config.FEAT_CONF_DIR, outfile), "w") as f:
        f.write(res)


def parse_args(parser):
    parser.add_option("-d", "--dim", default=1, type=int, dest="lsa_columns",
        help="lsa_columns")
    parser.add_option("-o", "--outfile", default="feature_conf_%s.py"%time_utils._timestamp(),
        type="string", dest="outfile", help="outfile")

    (options, args) = parser.parse_args()
    return options, args


def main(options):
    _create_feature_conf(lsa_columns=options.lsa_columns, outfile=options.outfile)


if __name__ == "__main__":
    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
