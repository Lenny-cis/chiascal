# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:54:06 2022

@author: linjianing
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, entropy
from itertools import (chain, combinations)
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

from .cutter import BinCutter
from ..utils.cut_merge import (
    cut_adjust, merge_arr_by_idx,
    arr_badrate_shape, calc_woe, calc_min_tol,
    woe_list2dict, apply_woe, split_na, concat_na)
from ..utils.progress_bar import make_tqdm_iterator


class BaseBinner(TransformerMixin, BaseEstimator):
    """探索性分箱."""

    def __init__(self, cut_cnt=50, min_PCT=0.025, min_n=None,
                 max_bin_cnt=6, I_min=3, U_min=4, cut_method='eqqt',
                 tolerance=0, precision=4, n_jobs=-1):
        self.cut_cnt = cut_cnt
        self.min_PCT = min_PCT
        self.min_n = min_n
        self.max_bin_cnt = max_bin_cnt
        self.I_min = I_min
        self.U_min = U_min
        self.cut_method = cut_method
        self.tolerance = tolerance
        self.precision = precision
        self.n_jobs = n_jobs
        self.bins_set = {}


def gen_comb_bins(crs, cut, I_min, U_min, variable_shape, max_bin_cnt,
                  tolerance, n_jobs):
    """生成全排列组合."""
    def comb_comb(hulkheads, loops):
        for loop in loops:
            yield combinations(hulkheads, loop)

    cross, na_arr = split_na(crs)
    minnum_bin_I = I_min
    minnum_bin_U = U_min
    vs = variable_shape
    maxnum_bin = max_bin_cnt
    tol = tolerance
    if 'U' not in vs:
        minnum_bin = minnum_bin_I
    elif 'I' not in vs and 'D' not in vs:
        minnum_bin = minnum_bin_U
    else:
        minnum_bin = min(minnum_bin_I, minnum_bin_U)
    rawnum_bin = cross.shape[0]
    bulkhead_list = list(range(1, cross.shape[0]))
    # 限定分组数的上下限
    maxnum_bulkhead_loops = max(rawnum_bin - minnum_bin, 0)
    minnum_bulkhead_loops = max(rawnum_bin - maxnum_bin, 0)
    loops_ = range(minnum_bulkhead_loops, maxnum_bulkhead_loops)
    bcs = comb_comb(bulkhead_list, loops_)
    # 多核并行计算
    var_bins = parallel_gen_bulkhead_bin(
        bcs, crs, minnum_bin_I, minnum_bin_U, vs, tol, cut, n_jobs)
    var_bin_dic = {k: v for k, v in enumerate(var_bins) if v is not None}
    return var_bin_dic


def parallel_gen_bulkhead_bin(bcs, arr, I_min, U_min,
                              variable_shape, tolerance, cut, n_jobs):
    """使用多核计算."""
    bcs = list(chain.from_iterable(bcs))
    tqdm_options = {'iterable': bcs, 'disable': False}
    progress_bar = make_tqdm_iterator(**tqdm_options)
    var_bins = Parallel(n_jobs=n_jobs)(delayed(gen_bulkhead_bin)(
        arr, merge_idxs, I_min, U_min, variable_shape, tolerance, cut)
        for merge_idxs in progress_bar)
    return var_bins


def gen_bulkhead_bin(arr, merge_idxs, I_min, U_min, variable_shape,
                     tolerance, cut):
    """计算变量分箱结果."""
    var_bin = gen_merged_bin(arr, merge_idxs, I_min, U_min,
                             variable_shape, tolerance)
    cut = cut_adjust(cut, merge_idxs)
    if var_bin is not None:
        var_bin.update({'cut': cut})
    return var_bin


def gen_merged_bin(arr, merge_idxs, I_min, U_min, variable_shape,
                   tolerance):
    """生成合并结果."""
    # 根据选取的切点合并列联表
    t_arr, na_arr = split_na(arr)
    merged_arr = merge_arr_by_idx(t_arr, merge_idxs)
    shape = arr_badrate_shape(merged_arr)
    # badrate的形状符合先验形状的分箱方式保留下来
    if pd.isna(shape) or (shape not in variable_shape):
        return
    elif shape in ['I', 'D']:
        if merged_arr.shape[0] < I_min:
            return
    else:
        if merged_arr.shape[0] < U_min:
            return
    masked_arr = concat_na(merged_arr, na_arr)
    detail = calc_woe(masked_arr)
    woes = detail['WOE']
    tol = calc_min_tol(woes[:-1])
    if tol <= tolerance:
        return
    chi, p, dof, expFreq =\
        chi2_contingency(merged_arr, correction=False)
    var_entropy = entropy(detail['all_num'][:-1])
    var_bin_ = {
        'detail': detail, 'flogp': -np.log(max(p, 1e-5)), 'tolerance': tol,
        'entropy': var_entropy, 'shape': shape, 'bin_cnt': len(merged_arr),
        'IV': detail['IV'].sum()
        }
    return var_bin_


class Combiner(BaseBinner):
    """全组合."""

    def __init__(self, cut_cnt=50, min_PCT=0.025, min_n=None,
                 max_bin_cnt=6, I_min=3, U_min=4, cut_method='eqqt',
                 tolerance=0, precision=4, n_jobs=-1, search_method='IV',
                 variable_shape='IDU'):
        super().__init__(cut_cnt, min_PCT, min_n, max_bin_cnt, I_min, U_min,
                         cut_method, tolerance, precision, n_jobs)
        self.variable_shape = variable_shape
        self.search_method = search_method
        self.raw_bins = {}

    def _gen_rawbins(self, X, y, **kwargs):
        """生产所有组合."""
        init_p = dict(self.get_params())
        del init_p['search_method']
        cutters = BinCutter(self.cut_cnt, self.min_PCT, self.min_n,
                            self.cut_method, self.precision, self.n_jobs)
        cutters.fit(X, y, **kwargs)
        for x_name in X.columns:
            x_p = {key: kwargs.get(key, {}).get(x_name, val)
                   for key, val in init_p.items()
                   if key not in ['cut_cnt', 'precision', 'cut_method',
                                  'min_PCT', 'min_n']}
            xcutter = cutters.split_set.get(x_name)
            if xcutter is None:
                return self
            crs = xcutter['cross']
            cut = xcutter['cut']
            xbin = gen_comb_bins(
                crs, cut, **x_p)
            self.raw_bins.update({x_name: xbin})
        return self

    def _fast_search_best(self, **kwargs):
        """单一目标搜索."""
        def _fast_search(bins, method):
            sort_bins = sorted(
                bins.items(), key=lambda x: [inflection_num[x[1]['shape']],
                                             -x[1][method], -x[1]['bin_cnt']])
            return sort_bins[0][1]

        inflection_num = {'I': 0, 'D': 0, 'U': 1}
        bins_set = self.raw_bins
        if len(bins_set) == 0:
            return self
        best_bins = Parallel(n_jobs=self.n_jobs)(
            delayed(_fast_search)(val, kwargs.get(key, self.search_method))
            for key, val in bins_set.items())
        self.bins_set = dict(zip(bins_set.keys(), best_bins))
        return self

    def fit(self, X, y, **kwargs):
        """最优分箱训练."""
        fit_params = {key: val for key, val in kwargs.items()
                      if key != 'search_method'}
        self._gen_rawbins(X, y, **fit_params)
        search_params = {key: val for key, val in kwargs.items()
                         if key == 'search_method'}
        self._fast_search_best(**search_params)
        return self

    def transform(self, X):
        """最优分箱转换."""
        cuts = {key: val['cut'] for key, val in self.bins_set.items()}
        woes = {key: woe_list2dict(val['detail']['WOE'])
                for key, val in self.bins_set.items()}
        cutters = BinCutter()
        cutters.set_cut(cuts)
        xcutted = cutters.transform(X)
        woe_dfs = Parallel(n_jobs=self.n_jobs)(
            delayed(apply_woe)(xcutted.loc[:, x_name], woes[x_name])
            for x_name in woes.keys())
        return pd.concat(woe_dfs, axis=1)
