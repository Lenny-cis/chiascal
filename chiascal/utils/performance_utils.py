# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:20:38 2021

@author: linjianing
"""


import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from autolrscorecard.utils.validate import param_in_validate
import autolrscorecard.variable_types.variable as vtype


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


def calc_iv(cross):
    """计算IV值，输入矩阵."""
    warnings.filterwarnings('ignore')
    mat = cross.values
    mat = np.where(mat == 0, 1e-5, mat)
    magc = mat.sum(axis=0)
    entropy = (mat[:, 1]/magc[1] - mat[:, 0]/magc[0])*np.log(
            (mat[:, 1]/magc[1]) / (mat[:, 0]/magc[0]))
    warnings.filterwarnings('default')
    return np.nansum(np.where((entropy == np.inf) | (entropy == -np.inf),
                              0, entropy))


def gen_ksseries(y, pred, pos_label=None):
    """生成KS及KS曲线相应元素.

    input
        y           实际flag
        pred        预测分数
        pos_label   标签被认为是响应，其他作为非响应
    """
    fpr, tpr, thrd = roc_curve(y, pred, pos_label=pos_label)
    eventNum = np.sum(y.to_list())
    allNum = len(y)
    nonEventNum = allNum - eventNum
    # KS对应的x点
    ksTile = (eventNum*tpr + nonEventNum*fpr)/allNum
    return np.max(tpr - fpr), tpr - fpr, ksTile


def calc_ks(y, pred, pos_label=None):
    """计算ks值."""
    return gen_ksseries(y, pred, pos_label)[0]


def calc_auc(y, pred, pos_label=None):
    """计算auc值."""
    fpr, tpr, thrd = roc_curve(y, pred, pos_label)
    return auc(fpr, tpr)
