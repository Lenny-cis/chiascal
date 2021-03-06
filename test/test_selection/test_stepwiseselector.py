# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:04:05 2022

@author: linjianing
"""

import os
import pandas as pd
import numpy as np
import statsmodels.stats.tests.test_influence

# test_module = statsmodels.stats.tests.test_influence.__file__
# cur_dir = cur_dir = os.path.abspath(os.path.dirname(test_module))

# file_name = "binary_constrict.csv"
# file_path = os.path.join(cur_dir, "results", file_name)
# df = pd.read_csv(file_path, index_col=0)

# ss = StepwiseSelector(value_in=1, value_out=-15)
import pandas as pd
from sklearn.pipeline import Pipeline

from chiascal.utils.comms import make_x_y
from chiascal.transform.combiner import Combiner
from chiascal.selection import StatsSelector, StepwiseSelector, TreeSelector


test_sample = pd.read_hdf(r'.\test\data\test_sample.h5', 'data')
test_sample = pd.read_excel(r'.\test\data\test_data.xlsx')
test_sample.set_index('客户名称', inplace=True)
# sample_X, sample_y = make_x_y(
#     test_sample, 'flag', **{'age_class': pd.CategoricalDtype(
#         categories=['A', 'B', 'C', 'D', 'E', 'F', 'G'], ordered=True)})

sample_X, sample_y = make_x_y(test_sample, 'flag')

SS = StatsSelector(n_jobs=1, noconcentration=0.05)
# TS = TreeSelector()
CB = Combiner()
STS = StepwiseSelector(value_in=1, value_out=2)
ppl = Pipeline([('statssl', SS),
                # ('treestep', TS),
                ('combinerbins', CB),
                ('stepwiseest', STS)])
ppl.fit(sample_X, sample_y)
# CB.update(sample_X, sample_y,
#           {'COUNT(profession)': [-np.inf, 1, 3, 15, 25, np.inf]})
# ppl.fit(sample_X, sample_y, combinerbins__refit=False, treestep__n_repeats=10)
pred = ppl.predict(sample_X)
ppl['stepwiseest'].final_model.summary2()
tr = ppl.transform(sample_X)
model_perf = ppl.score(sample_X, sample_y)
STS.set_score('Train', **ppl.score(sample_X, sample_y))
pd.DataFrame.from_dict(SS.pre_var_stats, orient='index')
STS.final_model.summary2()
