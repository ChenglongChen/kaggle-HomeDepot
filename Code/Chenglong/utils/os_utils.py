# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for os

"""

import os
import time
import shutil


def _gen_signature():
    # get pid and current time
    pid = int(os.getpid())
    now = int(time.time())
    # signature
    signature = "%d_%d" % (pid, now)
    return signature

def _create_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def _remove_files(files):
    for file in files:
        os.remove(file)

def _remove_dirs(dirs):
    for dir in dirs:
        shutil.rmtree(dir)
