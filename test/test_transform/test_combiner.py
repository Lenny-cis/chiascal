# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:47:48 2022

@author: linjianing
"""

import pandas as pd
from sklearn.pipeline import Pipeline

from chiascal.utils.comms import make_x_y
from chiascal.transform.combiner import Combiner
from chiascal.selection import StatsSelector


test_sample = pd.read_hdf(r'.\test\data\test_sample.h5', 'data')
sample_X, sample_y = make_x_y(
    test_sample, 'flag', **{'age_class': pd.CategoricalDtype(
        categories=['A', 'B', 'C', 'D', 'E', 'F', 'G'], ordered=True)})
ppl = Pipeline([('statssl', StatsSelector(n_jobs=1, noconcentration=0.5)),
               ('combinerbins', Combiner())])
ppl.fit(sample_X, sample_y)
tx = ppl.transform(sample_X)
ppl.steps[1][1].bins_set
