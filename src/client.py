#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cPickle as pickle
import pdb
import requests
import json

import jieba

#from api import get_simulator

host = 'http://localhost:1111/'
query = u"我來到北京清華大學"

print ', '.join(jieba.cut(query,cut_all=False))

firstpassdata = {'action':'firstpass','query':query}

response = requests.post(url=host+'query',data=firstpassdata)
print response.json()

