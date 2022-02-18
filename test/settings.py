# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:07:30 2021

@author: linjianing
"""


import pandas as pd
import os
path = os.path.split(os.path.realpath(__file__))[0]
file_path = os.path.join(path, r'行业分类.csv')
industry_map = pd.read_csv(file_path)
