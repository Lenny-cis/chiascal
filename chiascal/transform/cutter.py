# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:17:19 2022

@author: linjianing
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from sklearn.base import BaseEstimator, TransformerMixin

from .basebinner import BaseBinner
from .utils import (
    gen_cut, gen_cross, is_y_zero, merge_lowpct_zero, apply_cut_bin)


def gen_cross_cut(x, y, cut_cnt, cut_method, precision, min_PCT, min_n):
    """单变量训练."""
    if x.dropna().nunique() <= 1:
        return {}
    cut = gen_cut(x, n=cut_cnt, method=cut_method,
                  precision=precision)
    cross, cut = gen_cross(x, y, cut)
    if is_y_zero(cross):
        return {}
    mask=np.zeros_like(cross)
    mask[-1, :] = 1
    crs = np.ma.array(cross.to_numpy(), mask=mask)
    crs, cut = merge_lowpct_zero(crs, cut, min_PCT, min_n)
    return {'cross': crs, 'cut': cut}


class BinCutter(TransformerMixin, BaseEstimator):
    """分割器."""
    def __init__(self, cut_cnt=50, min_PCT=0.025, min_n=None,
                 cut_method='eqqt', n_jobs=-1):
        self.cut_cnt = cut_cnt
        self.min_PCT = min_PCT
        self.min_n = min_n
        self.cut_method = cut_method
        self.n_jobs = n_jobs
        self.spliter = {}

    def set_cut(self, cut_dict):
        orient_cut = self.get_cut()
        orient_cut.update(cut_dict)
        self.spliter = {key: {**self.spliter.get(key, {}), 'cut': val}
                        for key , val in orient_cut.items()}

    def get_cut(self):
        """分割器的切分点."""
        return {key: val['cut'] for key, val in self.split.items()}

    def fit(self, X, y, **kwargs):
        """分割X和y."""
        init_p = self.get_params()
        del init_p['n_jobs']
        var_bins = Parallel(n_jobs=self.n_jobs)(delayed(gen_cross_cut)(
            X.loc[:, x_name], y, **{**init_p, **kwargs.get('x_name', {})})
            for x_name in X.columns)
        self.spliter = dict(zip(X.columns.tolit(), var_bins))
        return self

    def transform(self, X):
        cuts = self.get_cut()
        cut_df = Parallel(n_jobs=self.n_jobs)(delayed(apply_cut_bin)(
            X.loc[:, x_name], cuts[x_name]) for x_name in cuts.keys())
        return pd.concat(cut_df, axis=1)
