# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for nlp

"""

import re


def _tokenize(text, token_pattern=" "):
    # token_pattern = r"(?u)\b\w\w+\b"
    # token_pattern = r"\w{1,}"
    # token_pattern = r"\w+"
    # token_pattern = r"[\w']+"
    if token_pattern == " ":
        # just split the text into tokens
        return text.split(" ")
    else:
        token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
        group = token_pattern.findall(text)
        return group
