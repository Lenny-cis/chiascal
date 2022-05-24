# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:42:52 2022

@author: linjianing
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
import logging


from ..utils.metrics import calc_iv
from ..utils.cut_merge import gen_cut, gen_cross


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


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
    iv_ = calc_iv(cross)
    return iv_


def var_stats(ser, y, thresholds):
    """变量统计信息."""
    missing_ratio_ = missing_ratio(ser)
    concentration_ratio_ = concentration_ratio(ser)
    num_unique_ = num_unique(ser)
    iv_ = iv(ser, y)
    res_ = {'nomissing': 1 - missing_ratio_,
            'noconcentration': 1-concentration_ratio_,
            'nunique': num_unique_,
            'IV': iv_}
    res_.update({'drop_reason': key for key, val in res_.items()
                 if val < thresholds.get(key)})
    return res_


class StatsSelector(TransformerMixin, BaseEstimator):
    """分箱前变量筛选."""

    def __init__(self, nomissing=0.05, noconcentration=0.05, nunique=0,
                 IV=0.01, n_jobs=1):
        self.nomissing = nomissing
        self.noconcentration = noconcentration
        self.nunique = nunique
        self.IV = IV
        self.n_jobs = n_jobs

    def fit(self, X, y, **kwargs):
        """筛选."""
        logger.info('Start {} fit'.format(self.__class__.__name__))
        init_p = dict(self.get_params())
        del init_p['n_jobs']
        # stats = []
        # for x_name in X.columns:
        #     stats.append(var_stats(X.loc[:, x_name], y, init_p))
        stats = Parallel(n_jobs=self.n_jobs)(
            delayed(var_stats)(X.loc[:, x_name], y, init_p)
            for x_name in X.columns)
        self.raw_var_stats = dict(zip(X.columns.tolist(), stats))
        self.pre_var_stats = {
            key: val for key, val in self.raw_var_stats.items()
            if pd.isna(val.get('drop_reason'))}
        return self

    def transform(self, X):
        """应用."""
        tran_x = self.pre_var_stats.keys()
        return X.loc[:, tran_x]

    def get_pre_IVs(self):
        return {key: val['IV'] for key, val in self.pre_var_stats.items()}
