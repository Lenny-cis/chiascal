# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:42:00 2022

@author: linjianing
"""

import pandas as pd
from chiascal.selection import StatsSelector

test_sample = pd.read_hdf(r'.\test\data\test_sample.h5', 'data')
sample_y = test_sample.loc[:, 'flag']
sample_X = test_sample.loc[:, test_sample.columns.difference([sample_y.name])]
statsselector = StatsSelector()
statsselector.fit(sample_X, sample_y)
