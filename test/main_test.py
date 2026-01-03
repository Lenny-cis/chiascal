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


src_path = os.path.split(os.getcwd())[0]
sys.path.append(src_path)
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


def multindex_filter(df, mi_dict={}):
    idx = pd.IndexSlice
    if not mi_dict:
        return df
    if len(set(list(mi_dict.keys())).difference(df.index.names))>0:
        raise ValueError('Wrong Index Name')
    idsl = df.index.nlevels * [slice(None)]
    for k, v in mi_dict.items():
        k_idx = list(df.index.names).index(k)
        if isinstance(v, (slice, list)):
            idsl[k_idx] = v
        else:
            idsl[k_idx] = [v]
    if isinstance(df, pd.DataFrame):
        return df.loc[idx[tuple(idsl)], :].copy()
    return df.loc[idx[tuple(idsl)]].copy()


test_data_file = os.path.join(src_path, 'test', 'data', 'test_data.xlsx')
test_data = pd.read_excel(test_data_file)
test_data = shuffle_test_data(test_data)

Y_COL = 'flag'
I_COL = '客户名称'

test_data.set_index(I_COL, inplace=True)
test_data = split_train_test_oot(test_data, Y_COL)
X_data, y_data = make_x_y(test_data, Y_COL)
X_trn = multindex_filter(X_data, {'split': "t00_Train"})
X_tst = multindex_filter(X_data, {'split': "t01_Test"})
X_oot = multindex_filter(X_data, {'split': "t02_OOT"})
y_trn = multindex_filter(y_data, {'split': "t00_Train"})
y_tst = multindex_filter(y_data, {'split': "t01_Test"})
y_oot = multindex_filter(y_data, {'split': "t02_OOT"})

ss1 = StatsSelector(nomissing=0.05, noconcentration=0.05, nunique=0, IV=0.01)
ps1 = PSISelector(psi=0.1, n_jobs=-1)
cg1 = CorrGraphSelector(corr=0.4, method='pearson')
cb1 = Combiner(
    cut_cnt=50, min_PCT=0.025, min_n=None,
    max_bin_cnt=6, I_min=3, U_min=4, cut_method='eqqt',
    tolerance=0.1, precision=6, n_jobs=-1, search_method='IV',
    variable_shape='IDU', modify=True)

sw1 = StepwiseSelector(
    p_value_in=0.05, p_value_out=0.1, criterion='aic',
    value_in=0.1, value_out=0.5)
pl = Pipeline([('statssel', ss1), ('psisel', ps1), ('corrgraphsel', cg1), ('combinersel', cb1), ('stepwisesel', sw1)])
pl.fit(X_trn, y_trn, psisel__rep_X=X_tst, corrgraphsel__iv_func=ss1.get_IVs, stepwisesel__verbose=False)
trn_score = pl.score(X_trn, y_trn)
tst_score = pl.score(X_tst, y_tst)
oot_score = pl.score(X_oot, y_oot)

print(trn_score.KS, trn_score.AUC)
print(tst_score.KS, tst_score.AUC)
print(oot_score.KS, oot_score.AUC)
# %%
text = '''
remiss,cell,smear,infil,li,blast,temp
1,.8,.83,.66,1.9,1.1,.996
1,.9,.36,.32,1.4,.74,.992
0,.8,.88,.7,.8,.176,.982
0,1,.87,.87,.7,1.053,.986
1,.9,.75,.68,1.3,.519,.98
0,1,.65,.65,.6,.519,.982
1,.95,.97,.92,1,1.23,.992
0,.95,.87,.83,1.9,1.354,1.02
0,1,.45,.45,.8,.322,.999
0,.95,.36,.34,.5,0,1.038
0,.85,.39,.33,.7,.279,.988
0,.7,.76,.53,1.2,.146,.982
0,.8,.46,.37,.4,.38,1.006
0,.2,.39,.08,.8,.114,.99
0,1,.9,.9,1.1,1.037,.99
1,1,.84,.84,1.9,2.064,1.02
0,.65,.42,.27,.5,.114,1.014
0,1,.75,.75,1,1.322,1.004
0,.5,.44,.22,.6,.114,.99
1,1,.63,.63,1.1,1.072,.986
0,1,.33,.33,.4,.176,1.01
0,.9,.93,.84,.6,1.591,1.02
1,1,.58,.58,1,.531,1.002
0,.95,.32,.3,1.6,.886,.988
1,1,.6,.6,1.7,.964,.99
1,1,.69,.69,.9,.398,.986
0,1,.73,.73,.7,.398,.986
'''
from io import StringIO
import statsmodels.api as sm
import scipy as sp
test_sas_data = pd.read_csv(StringIO(text), delimiter=',')
y = test_sas_data['remiss']
X = test_sas_data.drop(['remiss'], axis=1)
sw2 = StepwiseSelector(
    p_value_in=0.35, p_value_out=0.3, criterion='aic',
    value_in=0.1, value_out=0.5)
sw2.fit(X, y, verbose=True)
#X_const = sm.add_constant(X)
#slentry = 0.3
#slstay = 0.31
#included = ['const']
#clf_0_X = X_const.loc[:, included]
#clf_0 = sm.GLM(y, X_const.loc[:, ['const']],family=sm.families.Binomial())
#
#def forward(clf_0, X_const, slentry):
#    from dataclasses import dataclass, asdict
#    
#    @dataclass(order=True)
#    class chi2_score:
#        chi_square: float
#        p_value: float
#        name: str
#    
#    def ptest_score_chi_square(clf_res_0, new_col, X_const):
#        from scipy.stats import chi2
#        print(list(clf_res_0.params.index)+[new_col])
#        clf_1 = sm.GLM(y, X_const.loc[:, list(clf_res_0.params.index)+[new_col]],
#                       family=sm.families.Binomial())
#        p0 = clf_res_0.params
#        p0.loc[list(set(clf_1.exog_names)-set(clf_res_0.params.index))[0]] = 0
#        H = np.mat(clf_1.hessian(p0))
#        H_i = np.linalg.inv(H)
#        g = np.mat(clf_1.score(p0))
#        chi_s = (-g*H_i*g.T)[0, 0]
#        p = 1 - chi2.cdf(chi_s, 1)
#        return chi2_score(chi_s, p, new_col)
#    included = clf_0.exog_names
#    excluded = list(set(X_const.columns)-set(included))
#    clf_res_0 = clf_0.fit(disp=False)
#    AEEE = [ptest_score_chi_square(clf_res_0, x, X_const) for x in excluded]
#    AEEE.sort(reverse=True)
#    if AEEE[0].p_value >= slentry:
#        return clf_0
#
#    included.append(AEEE[0].name)
#    clf_1_X = X_const.loc[:, included]
#    clf_1 = sm.GLM(y, clf_1_X ,family=sm.families.Binomial())
#    return clf_1
#
#def backward(clf_0, slstay):
#    clf_res_0 = clf_0.fit(disp=False)
#    included = clf_0.exog_names
#    pvalues = clf_res_0.pvalues.iloc[1:].sort_values(ascending=False)
#    if pvalues.iloc[0] <= slstay:
#        return clf_0
#    
#    included.remove(pvalues.index[0])
#    clf_1_X = X_const.loc[:, included]
#    clf_1 = sm.GLM(y, clf_1_X ,family=sm.families.Binomial())
#    return clf_1
#    
#clf_0 = forward(clf_0, X_const, slentry)
#slstay = 0.2
#clf_0 = backward(clf_0, slstay)
#clf_0.fit(disp=False).summary2()
#
#pvalues = clf_res_0.pvalues.iloc[1:].sort_values(ascending=False)
#if pvalues.iloc[0] >= slstay:
#    included.remove(pvalues.index[0])
#
#mrs = sm.Logit(y, pd.DataFrame(
#            {'const': [1] * len(y)}, index=y.index)).fit(disp=False)
#mmr = mrs = sm.GLM(y, pd.DataFrame(
#            {'const': [1] * len(y)}, index=y.index),family=sm.families.Binomial())
#mrs = sm.GLM(y, pd.DataFrame(
#            {'const': [1] * len(y)}, index=y.index),family=sm.families.Binomial()).fit(disp=False)
#
#mrs.summary2()
#dir(mrs)
#mrs.resid_dev
#mrs.resid_generalized
#mrs.resid_pearson
#mrs.resid_response
#mmr1 = sm.GLM(
#    y, sm.add_constant(X.loc[:, ['li']]),family=sm.families.Binomial())
#mrs1 = sm.GLM(
#    y, sm.add_constant(X.loc[:, ['li']]),family=sm.families.Binomial())\
#    .fit(disp=False)
#rss0 = (mrs.resid_response**2).sum()
#rss1 = (mrs1.resid_response**2).sum()
#f = (rss0-rss1)/rss1*26
#f
#
#c1 = mrs.params
#c1.loc['infil'] = 0
#h = np.mat(mmr1.hessian(c1))
#h_1 = np.linalg.inv(h)
#g = np.mat(mmr1.score(c1))
#-g*h_1*g.T
#
#from dataclasses import dataclass, asdict
#
#@dataclass(order=True)
#class Employee:
#    name: str
#    salary: int
#    department: str = "Engineering"
#
#e1=Employee("Alice",85000)
#e2=Employee("Bob",92000)
#e3=Employee("Charlie",85000,"Marketing")
#print(e1<e2)#True-按字设顺序t比较(name,salary,department)
#print(sorted([e2,e1,e3]))
