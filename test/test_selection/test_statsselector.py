# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:42:00 2022

@author: linjianing
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from chiascal.selection import StatsSelector
from chiascal.collinear import CorrGraphSelector
from chiascal.utils.comms import make_x_y
test_sample = pd.read_hdf(r'.\test\data\test_sample.h5', 'data')
sample_y = test_sample.loc[:, 'flag'].map(lambda x: 1 if x else 0)
sample_X = test_sample.loc[:, test_sample.columns.difference([sample_y.name])]
# statsselector = StatsSelector(n_jobs=1, noconcentration=0.5)
# statsselector.fit(sample_X, sample_y)
sample_X, sample_y = make_x_y(
    test_sample, 'flag', **{'age_class': pd.CategoricalDtype(
        categories=['A', 'B', 'C', 'D', 'E', 'F', 'G'], ordered=True)})

SS = StatsSelector(n_jobs=1, noconcentration=0.05)
CGS = CorrGraphSelector()

ppl = Pipeline([('statssl', SS),
                ('corrstep', CGS)])
ppl.fit(sample_X, sample_y, corrstep__iv_func= SS.get_pre_IVs)
print(CGS.corr_vars)
# a = pd.DataFrame.from_dict(statsselector.pre_var_stats, orient='index')
# b = pd.DataFrame.from_dict(statsselector.raw_var_stats, orient='index')
