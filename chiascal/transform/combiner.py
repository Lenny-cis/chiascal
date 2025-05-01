# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:54:06 2022

@author: linjianing
"""
import numpy as np
import pandas as pd
from copy import copy
from scipy.stats import chi2_contingency, entropy
from itertools import (chain, combinations)
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
import logging
import inspect

from .cutter import BinCutter
from ..utils.cut_merge import (
    cut_adjust, merge_arr_by_idx, cut_to_interval,
    arr_badrate_shape, calc_woe, calc_min_tol,
    woe_list2dict, apply_woe, split_na, concat_na)
from ..utils.progress_bar import make_tqdm_iterator
from ..utils import FuncRunInfo, Report_tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


class BaseBinner(TransformerMixin, BaseEstimator):
    """探索性分箱."""

    def __init__(self, cut_cnt=50, min_PCT=0.025, min_n=None,
                 max_bin_cnt=6, I_min=3, U_min=4, cut_method='eqqt',
                 tolerance=0, precision=6, n_jobs=-1):
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
        self.output_vars = {}


def gen_comb_bins(crs, cut, I_min, U_min, variable_shape, max_bin_cnt,
                  tolerance, n_jobs, var, precision, modify):
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
    if rawnum_bin < minnum_bin:
        return {}
    bulkhead_list = list(range(1, cross.shape[0]))
    # 限定分组数的上下限
    maxnum_bulkhead_loops = max(rawnum_bin - minnum_bin, 1)
    minnum_bulkhead_loops = max(rawnum_bin - maxnum_bin, 0)
    loops_ = range(minnum_bulkhead_loops, maxnum_bulkhead_loops)
    bcs = chain.from_iterable(comb_comb(bulkhead_list, loops_))
    # 计算分箱
    var_bins = [gen_bulkhead_bin(crs, merge_idxs, minnum_bin_I,
                                 minnum_bin_U, vs, tol, cut, precision,
                                 modify)
                for merge_idxs in bcs]
    var_bin_dic = {k: v for k, v in enumerate(var_bins) if v is not None}
    return var_bin_dic


def gen_bulkhead_bin(arr, merge_idxs, I_min, U_min, variable_shape,
                     tolerance, cut, precision, modify):
    """计算变量分箱结果."""
    var_bin = gen_merged_bin(arr, merge_idxs, I_min, U_min,
                             variable_shape, tolerance, precision, modity)
    cut = cut_adjust(cut, merge_idxs)
    if var_bin is not None:
        cut = cut_adjust(cut, merge_idxs)
        var_bin.update({'cut': cut})
    return var_bin


def calc_details(masked_arr, threshold, precision, modity):
    """分箱细节."""
    detail = calc_woe(masked_arr, precision, modity)
    woes = detail['WOE']
    tol = calc_min_tol(woes[:-1])
    if tol< threshold:
        return
    merged_arr, na_arr = split_na(masked_arr)
    chi, p, dof, expFreq =\
        chi2_contingency(merged_arr, correction=False)
    var_entropy = entropy(detail['all_num'][:-1])
    return {'detail': detail, 'flogp': -np.log(max(p, 1e-5)), 'tolerance': tol,
            'entropy': var_entropy, 'bin_cnt': len(merged_arr),
            'IV': detail['IV'].sum()}


def gen_merged_bin(arr, merge_idxs, I_min, U_min, variable_shape,
                   tolerance, precision, modity):
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
    var_bin_ = calc_details(masked_arr, tolerance, precision, modify)
    if var_bin_ is None:
        return
    var_bin_.update({'shape': shape})
    return var_bin_


class Combiner(BaseBinner):
    """全组合."""

    def __init__(self, cut_cnt=50, min_PCT=0.025, min_n=None,
                 max_bin_cnt=6, I_min=3, U_min=4, cut_method='eqqt',
                 tolerance=0, precision=6, n_jobs=-1, search_method='IV',
                 variable_shape='IDU', modify=True):
        super().__init__(cut_cnt, min_PCT, min_n, max_bin_cnt, I_min, U_min,
                         cut_method, tolerance, precision, n_jobs)
        self.variable_shape = variable_shape
        self.search_method = search_method
        self.modity = modify
        self.input_vars = {}

    def _gen_rawbins(self, X, y, **kwargs):
        """生产所有组合."""
        def _gen_bins(i_cutter, **i_p):
            if i_cutter is None:
                return
            crs = i_cutter['cross']
            cut = i_cutter['cut']
            xbin = gen_comb_bins(crs, cut, **i_p)
            if len(xbin) <= 0:
                return
            return xbin
            
        init_p = dict(self.get_params())
        del init_p['search_method']
        cutters = BinCutter(self.cut_cnt, self.min_PCT, self.min_n,
                            self.cut_method, self.precision, self.n_jobs)
        cutters.fit(X, y, **kwargs)
        ban_p = ['cut_cnt', 'cut_method', 'min_PCT', 'min_n']
        tqdm_options = {'iterable': X.columns.tolist(), 'desc': 'Comb',
                        'disable': False}
        progress_bar = make_tqdm_iterator(**tqdm_options)
        res = Parallel(n_jobs=self.n_jobs)(delayed(_gen_bins)(
            cutters.split_set.get(x_name),
            **dict(**{key: kwargs.get(key, {}).get(x_name, val)
                      for key, val in init_p.items()
                      if key not in ban_p}, **{'var': x_name}))
            for x_name in progress_bar)
        self.input_vars = {key: val for key, val in
                           dict(zip(X.columns.tolist(), res)).items()
                           if val is not None}
        return self

    @FuncRunInfo(logger)
    def _fast_search_best(self, **kwargs):
        """单一目标搜索."""
        def _fast_search(bins, method):
            sort_bins = sorted(
                bins.items(), key=lambda x: [inflection_num[x[1]['shape']],
                                             -x[1][method], -x[1]['bin_cnt']])
            return sort_bins[0][1]

        inflection_num = {'I': 0, 'D': 0, 'U': 1}
        output_vars = self.input_vars
        if len(output_vars) == 0:
            return self
        tqdm_options = {'iterable': list(output_vars.keys()),
                        'desc': 'fast search',
                        'disable': False}
        progress_bar = make_tqdm_iterator(**tqdm_options)
        best_bins = Parallel(n_jobs=self.n_jobs)(
            delayed(_fast_search)(
                output_vars[key], kwargs.get(key, self.search_method))
            for key in progress_bar)
        self.output_vars = dict(zip(output_vars.keys(), best_bins))
        return self

    @FuncRunInfo(logger)
    def fit(self, X, y, refit=True, **kwargs):
        """最优分箱训练."""
        logger.info('Start {} fit'.format(self.__class__.__name__))
        if not refit:
            return self
        fit_params = {key: val for key, val in kwargs.items()
                      if key != 'search_method'}
        self._gen_rawbins(X, y, **fit_params)
        search_params = {key: val for key, val in kwargs.items()
                         if key == 'search_method'}
        self._fast_search_best(**search_params)
        return self

    @FuncRunInfo(logger)
    def transform(self, X):
        """最优分箱转换."""
        cuts = {key: val['cut'] for key, val in self.output_vars.items()
               if key in X.columns}
        woes = {key: woe_list2dict(val['detail']['WOE'])
                for key, val in self.output_vars.items() if key in X.columns}
        cutters = BinCutter()
        cutters.set_cut(cuts)
        xcutted = cutters.transform(X)
        woe_dfs = Parallel(n_jobs=self.n_jobs)(
            delayed(apply_woe)(xcutted[x_name], woes[x_name])
            for x_name in woes.keys())
        return pd.concat(woe_dfs, axis=1)

    @FuncRunInfo(logger)
    def update(self, X, y, new_cuts):
        """手动分箱."""
        def _calc_details(masked_arr, cut):
            merged_arr, na_arr = split_na(masked_arr)
            shape = arr_badrate_shape(merged_arr)
            var_bin_ = calc_details(
                masked_arr, self.tolerance, self.precision, self.modify)
            var_bin_.update({'shape': shape, 'cut': cut})
            return var_bin_

        cutters = BinCutter()
        cutters.set_cut(new_cuts, X, y)
        update_bins_lis = Parallel(n_jobs=self.n_jobs)(delayed(_calc_details)(
            cutters.allcross[key], cut) for key, cut in new_cuts.items())
        update_output_vars = dict(zip(new_cuts.keys(), update_bins_lis))
        self.output_vars.update(update_output_vars)
        return self

	def get_IVs(self):
        """获取IV列表."""
        return {key: val['IV'] for key, val in self.output_vars.items()}

	@property
    def rept(self):
        """Report DF."""
        def binset_to_tab(detail, cut):
            tdf = pd.DataFrame(detail)
            tdf.index = [-1 if x == tdf.shape[0]-1 else x for x in tdf.index]
            tdf.loc[:, 'Prop'] = tdf['all_num']/tdf['all_num'].sum()
            t_str_cut = cut_to_interval(cut)
            t_str_cut.update({-1: 'NAN'})
            tdf.loc[:,'bound'] = pd.Series(t_str_cut)
            return tdf[['bound', 'all_num', 'event_num', 'event_rate',
            			'Prop', 'WOE']]

		in_num = len(self.input_vars.keys())
        ou_num = len(self.output_vars.keys())
        summ = pd.DataFrame.from_dict({
        	'WOE': {'input': in_num, 'filter': in_num-ou_num,
        			'output': ou_num}},
        	orient='index')
        det = pd.concat({key: binset_to_tab(val['detail'], val['cut'])
        				 for key, val in self.output_vars.items()},
        				axis=0)
        return Report_tuple(summ, det)

	def sqlstmt(self, mvars):
        """Cut to woe sql代码."""
        sqlstmt = ',\n'.join([
        	f"case when {k} is null then {v['detail']['WOE'][-1]}"
        	+ "".join([f"\n\twhen {k}<={c} then {w}"
        			   for c, w in zip(v['cut'][1:], v['detail']['WOE'][:-2])])
        	+ f"\n\telse {v['detail']['WOE'][-2]} end as {k}_woe"
        	for k, v in self.output_vars.items() if k in mvars])
        return sqlstmt
