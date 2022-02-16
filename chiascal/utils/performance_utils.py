# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:20:38 2021

@author: linjianing
"""


import numpy as np
import pandas as pd
from .validate import param_in_validate


def gen_cut_summ(ser, n=10, mthd='eqqt', precision=4):
    """
    生成连续变量的切分点.

    若序列中只有一个值，返回字符串"N_CUT ERROR"
    input
        ser         序列
        n           切点个数
        mthd        切分方式
            eqqt    等频
            eqdist  等距
    """
    param_in_validate(mthd, ['eqqt', 'eqdist'], '使用了mthd={0}, 仅支持mthd={1}')
    rcut = list(sorted(ser.dropna().unique()))
    if len(rcut) <= 2:
        return [-np.inf, np.inf]
    if mthd == 'eqqt':
        cut = list(np.unique(pd.qcut(ser, n, retbins=True, duplicates='drop')
                             [1].round(precision)))
    elif mthd == 'eqdist':
        cut = list(np.unique(pd.cut(ser, n, retbins=True, duplicates='drop')[1]
                   .round(precision)))
    cut[0] = -np.inf
    cut[-1] = np.inf
    return cut


def gen_cut_count(ser):
    """生成计数变量的切分点."""
    rcut = list(sorted(ser.dropna().unique()))
    if len(rcut) <= 1:
        return [-np.inf, np.inf]
    rcut[0] = -np.inf
    rcut[-1] = np.inf
    return rcut


def gen_cut_discrete(ser):
    """生成分类变量切分点."""
    return {i: val for i, val in enumerate(ser.categories)}
