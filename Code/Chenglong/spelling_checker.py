# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: spelling checker

"""

import gc
import time
import multiprocessing
from collections import Counter

import re, regex
import nltk
import difflib

import config
import google_spelling_checker_dict


class DictSpellingChecker:
    def __init__(self, spelling_checker_dict):
        self.spelling_checker_dict = spelling_checker_dict

    def correct(self, query):
        return self.spelling_checker_dict.get(query, query)


class GoogleQuerySpellingChecker(DictSpellingChecker):
    def __init__(self):
        super().__init__(google_spelling_checker_dict.spelling_checker_dict)


class PeterNovigSpellingChecker:
    """
    How to Write a Spelling Corrector
    http://norvig.com/spell-correct.html
    """
    def __init__(self, words, error_correction_pairs=None, exclude_stopwords=False, cutoff=0.8):
        """
        error_correction_pairs: a list of word error pairs in the form [("w1", "c1"), ("w2", "c2")]
        this is for estimating P(w|c)
        """
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.whitespace = " "
        self.digit = "0123456789"
        self.atom = self.alphabet + self.whitespace + self.digit
        self.exclude_stopwords = exclude_stopwords
        self.words = words
        self.error_correction_pairs = error_correction_pairs
        if self.exclude_stopwords:
            self.words = [w for w in self.words if w not in config.STOP_WORDS]
            self.error_correction_pairs = [(e,c) for (e,c) in self.error_correction_pairs 
                                            if (e not in config.STOP_WORDS and c not in config.STOP_WORDS)]
        self.P_c = self.get_P_c(self.words)
        self.P_w_given_c = self.get_P_w_given_c(self.error_correction_pairs)
        self.cutoff = cutoff
        self.words_uniq = list(set(self.words))

    # to make it pickable
    def get_P_c(self, words):
        model = {}
        # model = defaultdict(lambda : 1)
        for w in words:
            # if w not in model, initilize it as 1
            # else return the value in model
            model[w] = model.get(w, 1) + 1
            # model[w] += 1
        return model

    # to make it pickable
    def get_P_w_given_c(self, error_correction_pairs):
        model = {}
        # model = defaultdict(lambda : defaultdict(lambda : 1))
        for e,c in error_correction_pairs:
            if not e in model:
                model[e] = {}
            model[e][c] = model.get(e, {}).get(c, 1) + 1
            # model[e][c] += 1
        return model

    def edits1(self, word):
       splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
       deletes    = [a + b[1:] for a, b in splits if b]
       transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
       replaces   = [a + c + b[1:] for a, b in splits for c in self.atom if b]
       inserts    = [a + c + b     for a, b in splits for c in self.atom]
       return set(deletes + transposes + replaces + inserts)

    def known(self, words):
        return set(w for w in words if w in self.P_c)

    # v1
    def known_edits1_v1(self, word):
        return set(e1 for e1 in self.edits1(word) if e1 in self.P_w_given_c.get(word, {}))

    def known_edits2_v1(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.P_w_given_c.get(word, {}))

    def known_edits3_v1(self, word):
        return set(e3 for e1 in self.edits1(word) for e2 in self.edits1(e1) for e3 in self.edits1(e2) if e3 in self.P_w_given_c.get(word, {}))        

    def correct_major_v1(self, word):
        # original code
        # candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        # # we simply consider all the candidates and take the one with maximum count
        # # this is equal to just look at P(c)
        # candidates = self.known([word]) | self.known(self.edits1(word)) | self.known_edits2(word) | set([word])
        # word
        corrected_word = word
        _p = -1
        # known
        candidates = self.known([word])
        if candidates:
            corrected_word = word
        else:
            # known edits1
            candidates = self.known_edits1_v1(word)
            if candidates:
                for c in candidates:
                    p = self.P_c.get(c, 1) * self.P_w_given_c.get(word, {}).get(c, 1)
                    if p > _p:
                        corrected_word = c
                        _p = p
            else:
                # known edits2
                candidates = self.known_edits2_v1(word)
                if candidates:
                    for c in candidates:
                        p = self.P_c.get(c, 1) * self.P_w_given_c.get(word, {}).get(c, 1)
                        if p > _p:
                            corrected_word = c
                            _p = p
                # else:
                #     # known edits3
                #     candidates = self.known_edits3_v1(word)
                #     if candidates:
                #         for c in candidates:
                #             p = self.P_c.get(c, 1) * self.P_w_given_c.get(word, defaultdict(lambda :1)).get(c, 1)
                #             if p > _p:
                #                 corrected_word = c
                #                 _p = p
        return corrected_word

    # v2
    def known_edits1_v2(self, word):
        return set(e1 for e1 in self.edits1(word) if e1 in self.P_c)

    def known_edits2_v2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.P_c)

    def known_edits3_v2(self, word):
        return set(e3 for e1 in self.edits1(word) for e2 in self.edits1(e1) for e3 in self.edits1(e2) if e3 in self.P_c)        

    def correct_major_v2(self, word):
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2_v2(word) or [word]
        return max(candidates, key=self.P_c.get)

    def correct_backup(self, word):
        try:
            corrected_word = difflib.get_close_matches(word, self.P_w_given_c[word], n=1, cutoff=self.cutoff)
            if len(corrected_word) > 0:
                return corrected_word[0]
            else:
                return word
        except:
            return word

    def correct(self, word):
        corrected_word = self.correct_major_v1(word)
        count = self.P_c.get(corrected_word, 0)
        if count < 1:
            corrected_word = self.correct_major_v2(word)
            count = self.P_c.get(corrected_word, 0)
            if count < 1:
                return self.correct_backup(word)
            else:
                return corrected_word
        else:
            return corrected_word


class AutoSpellingChecker:
    """
    https://www.kaggle.com/hiendang/crowdflower-search-relevance/auto-correct-query
    """
    def __init__(self, dfAll, exclude_stopwords=False, skip_digit=True,
                    cutoff=0.9, min_len=5, n_jobs=config.AUTO_SPELLING_CHECKER_N_JOBS):
        self.dfAll = dfAll.copy().sort_values("search_term")
        self.exclude_stopwords = exclude_stopwords
        self.skip_digit = skip_digit
        self.cutoff = cutoff
        self.min_len = min_len
        self.n_jobs = n_jobs

        self.get_dfs_words(["search_term", "product_title", "product_description", "product_attribute"])
        self.words = self.build_word_corpus()
        self.error_correction_pairs = self.build_error_correction_pairs()
        self.spelling_checker = self.build_spelling_checker()
        self.query_correction_map = self.build_query_correction_map()

    # build query neighbors
    def get_words(self, doc):
        words = re.findall(r'[\'\"\w]+', doc)
        return words

    def get_valid_unigram_words(self, words):
        _words = []
        for word in words:
            if len(word) >= self.min_len:
                if (not self.exclude_stopwords) or (word not in config.STOP_WORDS):
                    if (not self.skip_digit) or (len(re.findall(re.compile("\d+"), word)) == 0):
                        _words.append(word)
        return _words

    def get_valid_bigram_words(self, words):
        _words = []
        for i in nltk.bigrams(words):
            if (len(i[0]) >= self.min_len) and (len(i[1]) >= self.min_len):
                if (not self.exclude_stopwords) or ((i[0] not in config.STOP_WORDS) and (i[1] not in config.STOP_WORDS)):
                    if (not self.skip_digit) or ((len(re.findall(re.compile("\d+"), i[0])) == 0) and (len(re.findall(re.compile("\d+"), i[1])) == 0)):
                        _words.append(" ".join(i))
        return _words

    def get_str_words(self, doc):
        words = self.get_words(doc)
        words_unigram = self.get_valid_unigram_words(words)
        words_bigram = self.get_valid_bigram_words(words)
        words = words_unigram + words_bigram
        if len(words) == 0:
            words = [""]
        return words

    def get_dfs_words(self, columns):
        for col in columns:
            self.dfAll["%s_words"%col] = self.dfAll[col].apply(self.get_str_words)

    # for estimating P(c)
    def build_word_corpus(self):
        columns = ["product_title", "product_description", "product_attribute"]
        columns = [col+"_words" for col in columns if col in self.dfAll.columns]
        words = []
        for col in columns:
            # flatten it which is a list of list
            this_words = sum(list(self.dfAll[col]), [])
            words.extend( this_words )
        return words

    # for estimating P(w|c)
    def build_error_correction_pairs(self):
        columns = ["product_title", "product_description"]
        pairs = []
        for col in columns:
            for query,title in zip(self.dfAll["search_term_words"], self.dfAll["%s_words"%col]):
                for word in query:
                    pairs.extend( [(word,c) for c in title] )
        return pairs

    def build_spelling_checker(self):
        checker = PeterNovigSpellingChecker(self.words, self.error_correction_pairs,
                    self.exclude_stopwords, self.cutoff)
        return checker

    def autocorrect_query(self, query):
        # correct bigram
        bigram_corrector = {}
        words_bigram = self.get_valid_bigram_words(query.split(" "))
        for word in words_bigram:
            corrected_word = self.spelling_checker.correct(word)
            if word != corrected_word:
                bigram_corrector[word] = corrected_word

        for k,v in bigram_corrector.items():
            pattern = regex.compile(r"(?<=\W|^)%s(?=\W|$)"%k)
            query = regex.sub(pattern, v, query)

        # correct unigram
        corrected_query = []
        for word in query.split(" "):
            if len(word) < self.min_len:
                corrected_query.append(word)
            elif self.exclude_stopwords and word in config.STOP_WORDS:
                corrected_query.append(word)
            elif self.skip_digit and len(re.findall(re.compile("\d+"), word)):
                corrected_query.append(word)
            else:
                corrected_word = self.spelling_checker.correct(word)
                if len(corrected_word) < self.min_len:
                    corrected_query.append(word)
                else:
                    corrected_query.append( corrected_word )
        return " ".join(corrected_query)

    def build_query_correction_map(self):
        queries = list(set(self.dfAll["search_term"]))
        if self.n_jobs == 1:
            corrected_queries = []
            for query in queries:
                corrected_queries.append( self.autocorrect_query(query) )
        else:
            p = multiprocessing.Pool(self.n_jobs)
            corrected_queries = p.imap(self.autocorrect_query, queries)
        query_correction_map = dict(zip(queries, corrected_queries))
        return query_correction_map

    def correct(self, query):
        return self.query_correction_map.get(query, query)

    def save_query_correction_map(self, fname):
        with open(fname, "w") as f:
            f.write("%30s \t %30s\n"%("original query", "corrected query"))
            for query,corrected_query in sorted(self.query_correction_map.items()):
                if query != corrected_query:
                    res = "%30s \t %30s\n"%(query, corrected_query)
                    f.write(res)
