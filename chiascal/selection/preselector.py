# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:42:52 2022

@author: linjianing
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed


from ..utils.performance_utils import calc_iv
from ..utils.transform_utils import gen_cut, gen_cross

def missing_ratio(ser):
    """计算缺失率."""
    return ser.isna().sum() / len(ser)


def concentration_ratio(ser):
    """计算集中度."""
    if all(ser.isna()):
        return np.nan
    return ser.value_counts().iloc[0] / ser.count()


def num_unique(ser):
    """不同值的个数."""
    return ser.nunique()


def iv(ser, y):
    """变量IV."""
    cut = gen_cut(ser, n=20, mthd='eqqt', precision=4)
    cross, cut = gen_cross(ser, y, cut)
    iv = calc_iv(cross)
    return iv


def var_stats(ser, y):
    missing_ratio_ = missing_ratio(ser)
    concentration_ratio_ = concentration_ratio(ser)
    num_unique_ = num_unique(ser)
    iv_ = iv(ser, y)
    return {'nomissing': 1 - missing_ratio_,
            'nonconcentration': 1-concentration_ratio_,
            'nunique': num_unique_,
            'IV': iv_}


class PreSelecter(TransformerMixin, BaseEstimator):
    def __init__(self, nomissing=0.05, nonconcentration = 0.05, nunique = 0,
                 IV = 0.01, n_jobs=-1):
        self.nomissing = nomissing
        self.nonconcentration = nonconcentration
        self.nunique = nunique
        self.IV = IV
        self.n_jobs = n_jobs

    def fit(self, X, y, **kwargs):
        init_p = dict(self.get_params())
        del init_p['n_jobs']
        stats = Parallel(n_jobs=self.n_jobs)(
            delayed(var_stats)(X.loc[:, x_name], y) for x_name in X.columns)
        self.raw_var_stats = dict(zip(X.columns.tolist(), stats))
        self.pre_var_stats = {
            key: val for key, val in self.var_stats.items()
            if all([val[wkey] >= wval.get(key, init_p[wkey])
                    for wkey, wval in kwargs.items()])}
