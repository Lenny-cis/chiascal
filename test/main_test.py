# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 14:36:56 2025

@author: Lenny
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline

sys.path.append(r'D:\risk_code\chiascal')
from chiascal.utils import make_x_y
from chiascal.selection import StatsSelector, PSISelector, StepwiseSelector
from chiascal.transform import Combiner
from chiascal.collinear import CorrGraphSelector


def shuffle_test_data(df):
    months = [x.strftime('%Y-%m') for x in pd.date_range('2022-01', '2022-12', freq='MS')]
    tdf = shuffle(df)
    tdf.loc[:, 'apply_month'] = np.random.choice(months, df.shape[0])
    return tdf


def split_train_test_oot(df, y_col):
    tdf = df.copy()
    tdf.loc[tdf['apply_month']>='2022-10', 'split'] = 't02_OOT'
    month_y = tdf.loc[tdf['split']!='t02_OOT', ['apply_month', y_col]]
    train_, test_ = train_test_split(
        tdf.loc[tdf['split']!='t02_OOT', :],
        test_size=0.3, random_state=2024, stratify=month_y)
    tdf.loc[train_.index, 'split'] = 't00_Train'
    tdf.loc[test_.index, 'split'] = 't01_Test'
    tdf.set_index(['apply_month', 'split'], append=True, inplace=True)
    return tdf


test_data_file = os.path.abspath(r'D:\risk_code\chiascal\test\data\test_data.xlsx')
test_data = pd.read_excel(test_data_file)
test_data = shuffle_test_data(test_data)

Y_COL = 'flag'
I_COL = '客户名称'

test_data.set_index(I_COL, inplace=True)
test_data = split_train_test_oot(test_data, Y_COL)
X_data, y_data = make_x_y(test_data, Y_COL)
X_trn = X_data.loc[pd.IndexSlice[:, :, "t00_Train"]]
X_tst = X_data.loc[pd.IndexSlice[:, :, "t01_Test"]]
X_oot = X_data.loc[pd.IndexSlice[:, :, "t02_OOT"]]
y_trn = y_data.loc[pd.IndexSlice[:, :, "t00_Train"]]
y_tst = y_data.loc[pd.IndexSlice[:, :, "t01_Test"]]
y_oot = y_data.loc[pd.IndexSlice[:, :, "t02_OOT"]]

ss1 = StatsSelector(nomissing=0.05, noconcentration=0.05, nunique=0, IV=0.01)
ps1 = PSISelector(psi=0.1, n_jobs=-1)
cg1 = CorrGraphSelector(corr=0.4, method='spearman')
cb1 = Combiner(
    cut_cnt=50, min_PCT=0.025, min_n=None,
    max_bin_cnt=6, I_min=3, U_min=4, cut_method='eqqt',
    tolerance=0.1, precision=6, n_jobs=-1, search_method='IV',
    variable_shape='IDU', modify=True)

sw1 = StepwiseSelector(
    p_value_in=0.05, p_value_out=0.01, criterion='aic',
    value_in=0.1, value_out=0.5)
pl = Pipeline([('statssel', ss1), ('psisel', ps1), ('corrgraphsel', cg1), ('combinersel', cb1), ('stepwisesel', sw1)])
pl.fit(X_trn, y_trn, psisel__rep_X=X_tst, corrgraphsel__iv_func=ss1.get_IVs)
trn_score = pl.score(X_trn, y_trn)
tst_score = pl.score(X_tst, y_tst)
oot_score = pl.score(X_oot, y_oot)

print(trn_score.KS, trn_score.AUC)
print(tst_score.KS, tst_score.AUC)
print(oot_score.KS, oot_score.AUC)
