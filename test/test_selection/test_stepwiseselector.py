# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:04:05 2022

@author: linjianing
"""

import os
import pandas as pd
import statsmodels.stats.tests.test_influence

test_module = statsmodels.stats.tests.test_influence.__file__
cur_dir = cur_dir = os.path.abspath(os.path.dirname(test_module))

file_name = "binary_constrict.csv"
file_path = os.path.join(cur_dir, "results", file_name)
df = pd.read_csv(file_path, index_col=0)

ss = StepwiseSelector(value_in=1, value_out=-15)
ss.fit(df[['rate', 'volumne', 'log_rate', 'log_volumne']], df['constrict'])
pred_p = ss.predict(df.loc[:, ['rate', 'volumne', 'log_rate', 'log_volumne']])
smodel = sm.Logit(df['constrict'], pd.DataFrame({'const': [1] *len(df['constrict'])}, index=df['constrict'].index)).fit()
smodel = sm.Logit(df['constrict'], df[['rate', 'volumne', 'log_rate', 'log_volumne']]).fit()
smodel.summary2()
smodel = sm.Logit(df['constrict'], df[['const', 'volumne', 'log_rate']]).fit()
smodel.summary2()
