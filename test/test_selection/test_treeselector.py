# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:25:38 2022

@author: Lenny
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from chiascal.utils.comms import make_x_y
from chiascal.transform.combiner import Combiner
from chiascal.selection import StatsSelector, StepwiseSelector, TreeSelector


import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
rng = np.random.RandomState(seed=42)
X["random_cat"] = rng.randint(3, size=X.shape[0])
X["random_num"] = rng.randn(X.shape[0])

categorical_columns = ["pclass", "sex", "embarked", "random_cat"]
numerical_columns = ["age", "sibsp", "parch", "fare", "random_num"]

X = X[categorical_columns + numerical_columns]
X.loc[:, categorical_columns] = X.loc[:, categorical_columns].astype(pd.CategoricalDtype())
y = y.astype(int)



test_sample = pd.read_hdf(r'.\test\data\test_sample.h5', 'data')
sample_X, sample_y = make_x_y(
    test_sample, 'flag', **{'age_class': pd.CategoricalDtype(
        categories=['A', 'B', 'C', 'D', 'E', 'F', 'G'], ordered=True)})
SS = StatsSelector(n_jobs=1, noconcentration=0.05)
TS = TreeSelector()
ppl = Pipeline([('statsstep', SS), ('treestep', TS)])
ppl.fit(sample_X, sample_y, treestep__n_repeats=10)
