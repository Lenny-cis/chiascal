# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:17:19 2022

@author: linjianing
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
import logging

from ..utils.cut_merge import (
    gen_cut, gen_cross, is_y_zero, merge_lowpct_zero, apply_cut_bin)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


def gen_cross_cut(x, y, cut_cnt, cut_method, precision, min_PCT, min_n):
    """单变量训练."""
    if x.dropna().nunique() <= 1:
        return {}
    cut = gen_cut(x, n=cut_cnt, method=cut_method,
                  precision=precision)
    cross, cut = gen_cross(x, y, cut)
    if is_y_zero(cross):
        return {}
    mask = np.zeros_like(cross)
    mask[-1, :] = 1
    crs = np.ma.array(cross.to_numpy(), mask=mask)
    crs, cut = merge_lowpct_zero(crs, cut, min_PCT, min_n)
    return {'cross': crs, 'cut': cut}


def gen_cross_from_cut(x, y, cut):
    """根据手动切分点生成spliter."""
    cross, cut = gen_cross(x, y, cut)
    mask = np.zeros_like(cross)
    mask[-1, :] = 1
    crs = np.ma.array(cross.to_numpy(), mask=mask)
    return {'cross': crs, 'cut': cut}


class BinCutter(TransformerMixin, BaseEstimator):
    """分割器."""

    def __init__(self, cut_cnt=50, min_PCT=0.025, min_n=None,
                 cut_method='eqqt', precision=4, n_jobs=1):
        self.cut_cnt = cut_cnt
        self.min_PCT = min_PCT
        self.min_n = min_n
        self.cut_method = cut_method
        self.n_jobs = n_jobs
        self.precision = precision
        self.split_set = {}

    def set_cut(self, cut_dict, X=None, y=None):
        """指定切分点."""
        orient_cut = self.allcut
        orient_cut.update(cut_dict)
        if X is not None and y is not None:
            new_crslis = Parallel(n_jobs=self.n_jobs)(delayed(
                gen_cross_from_cut)(X.loc[:, x_name], y, cut)
                for x_name, cut in cut_dict.items())
            new_split = dict(zip(cut_dict.keys(), new_crslis))
        else:
            new_split = {key: {**self.split_set.get(key, {}), 'cut': val}
                         for key, val in orient_cut.items()}
        self.split_set.update(new_split)
        return self

    @property
    def allcut(self):
        """分割器的切分点."""
        return {key: val['cut'] for key, val in self.split_set.items()}

    @property
    def allcross(self):
        """切分器的列联表."""
        return {key: val['cross'] for key, val in self.split_set.items()}

    def fit(self, X, y, **kwargs):
        """分割X和y."""
        init_p = dict(self.get_params())
        del init_p['n_jobs']
        var_bins = Parallel(n_jobs=self.n_jobs)(delayed(gen_cross_cut)(
            X.loc[:, x_name], y,
            **{key: kwargs.get(x_name, {}).get(key, val)
                for key, val in init_p.items()})
            for x_name in X.columns)
        self.split_set = dict(zip(X.columns.tolist(), var_bins))
        return self

    def transform(self, X):
        """应用分割."""
        cuts = self.allcut
        cut_df = Parallel(n_jobs=self.n_jobs)(delayed(apply_cut_bin)(
            X.loc[:, x_name], cuts[x_name]) for x_name in cuts.keys())
        return pd.concat(cut_df, axis=1)
