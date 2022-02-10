# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:54:06 2022

@author: linjianing
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, entropy
from itertools import (chain, product, combinations)
from joblib import Parallel, delayed, dump, load

from .basebinner import BaseBinner
from .utils import (
    gen_cut, gen_cross, is_y_zero, cut_adjust, merge_arr_by_idx, arr_badrate_shape, calc_woe, calc_min_tol, apply_woe, apply_cut_bin, merge_lowpct_zero,
    make_tqdm_iterator, parallel_gen_var_bin,
    woe_list2dict, gen_var_bin)


def gen_comb_bins(crs, cut, I_min, U_min, var_shape, max_bin_cnt,
                  tolerance, n_jobs):
    def comb_comb(hulkheads, loops):
        for loop in loops:
            yield combinations(hulkheads, loop)

    cross = crs.copy()
    minnum_bin_I = I_min
    minnum_bin_U = U_min
    vs = var_shape
    maxnum_bin = max_bin_cnt
    tol = tolerance
    if 'U' not in vs:
        minnum_bin = minnum_bin_I
    elif 'I' not in vs and 'D' not in vs:
        minnum_bin = minnum_bin_U
    else:
        minnum_bin = min(minnum_bin_I, minnum_bin_U)
    rawnum_bin = cross.shape[0]
    hulkhead_list = list(range(1, cross.shape[0]))
    # 限定分组数的上下限
    maxnum_hulkhead_loops = max(rawnum_bin - minnum_bin, 0)
    minnum_hulkhead_loops = max(rawnum_bin - maxnum_bin, 0)
    loops_ = range(minnum_hulkhead_loops, maxnum_hulkhead_loops)
    bcs = comb_comb(hulkhead_list, loops_)
    # 多核并行计算
    var_bins = parallel_gen_var_bin(
        bcs, crs, minnum_bin_I, minnum_bin_U, vs, tol, cut, n_jobs)
    var_bin_dic = {k: v for k, v in enumerate(var_bins) if v is not None}
    return var_bin_dic


def varbin(x, y, var_type, cut_cnt, cut_method, precision, threshold_PCT,
           threshold_n, I_min, U_min, var_shape, max_bin_cnt, tolerance):
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
    crs, cut = merge_lowpct_zero(crs, cut, threshold_PCT, threshold_n)
    bin_dic = gen_comb_bins(crs, cut, I_min, U_min, var_shape,
                            max_bin_cnt, tolerance)
    return bin_dic


def parallel_gen_var_bin(bcs, arr, I_min, U_min,
                         variable_shape, tolerance, cut, n_jobs):
    """使用多核计算."""
    bcs = list(chain.from_iterable(bcs))
    tqdm_options = {'iterable': bcs, 'disable': False}
    progress_bar = make_tqdm_iterator(**tqdm_options)
    var_bins = Parallel(n_jobs=n_jobs)(delayed(gen_var_bin)(
        arr, merge_idxs, I_min, U_min, variable_shape, tolerance, cut)
        for merge_idxs in progress_bar)
    return var_bins


def gen_var_bin(arr, merge_idxs, I_min, U_min, variable_shape,
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
    t_arr = np.ma.compress_rows(arr)
    na_arr = np.expand_dim(arr.data[arr.mask], axis=0)
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
    masked_arr = np.concatenate((merged_arr, na_arr), axis=0)
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

    def __init__(self, cut_cnt=50, thrd_PCT=0.025, thrd_n=None,
                 max_bin_cnt=6, I_min=3, U_min=4, cut_mthd='eqqt',
                 tolerance=0, n_jobs=-1):
        super(cut_cnt, thrd_PCT, thrd_n, max_bin_cnt, I_min, U_min, cut_mthd,
              tolerance, n_jobs)

    def fit(self, X, y, **kwargs):
        """训练."""
        for x_name in X.columns:
            pass
