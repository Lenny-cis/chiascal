# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:28:43 2022

@author: linjianing
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


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
    fpr, tpr, thrd = roc_curve(y, pred, pos_label=pos_label)
    return auc(fpr, tpr)


def gen_gaintable(y, pred, bins=20, prob=True):
    """生成gaintable."""
    t_df = pd.DataFrame({'pred': pred, 'flag': y})
    t_df.loc[:, 'cut'] = pd.qcut(
        t_df.loc[:, 'pred'], bins, labels=False, duplicates='drop')
    if prob:
        t_df.loc[:, 'cut'] = t_df.loc[:, 'cut'].max() - t_df.loc[:, 'cut']
    total_good_num, total_bad_num = y.value_counts()
    num_df = t_df.groupby(['cut', 'flag']).size().unstack().rename(
        columns={0: 'good_num', 1: 'bad_num'})
    num_df.loc[:, 'total'] = num_df.sum(axis=1)
    cum_num_df = num_df.cumsum()
    rev_cum_num_df = num_df.sort_values('cut', ascending=False).cumsum()
    rev_cum_num_df = rev_cum_num_df.assign(
        rev_lift=lambda x: x['good_num'] / total_good_num / (
            x['total'] / len(t_df)))
    num_df.loc[:, 'bad_rate'] = num_df.loc[:, 'bad_num']\
        / num_df.loc[:, 'total']
    cum_num_df.loc[:, 'bad_rate'] = cum_num_df.loc[:, 'bad_num']\
        / cum_num_df.loc[:, 'total']
    cum_num_df.columns = cum_num_df.columns.map(lambda x: 'cum_' + x)
    score_df = t_df.groupby(['cut'])['pred'].agg(['min', 'max'])\
        .rename(columns={'min': 'min_score', 'max': 'max_score'})
    score_df.loc[:, 'score_range'] = score_df.apply(
            lambda x: pd.Interval(x['min_score'], x['max_score']), axis=1)
    res_df = pd.concat([score_df, num_df, cum_num_df,
                        rev_cum_num_df.loc[:, ['rev_lift']]], axis=1)
    res_df.loc[:, 'gain'] = res_df.loc[:, 'cum_bad_num'] / total_bad_num
    res_df.loc[:, 'lift'] = res_df.loc[:, 'gain']/((res_df.loc[:, 'cum_total'])
                                                   / (len(t_df)))
    res_df.loc[:, 'ks'] = np.abs(
            res_df.loc[:, 'cum_good_num'] / total_good_num
            - res_df.loc[:, 'cum_bad_num'] / total_bad_num)
    return res_df.loc[:, ['score_range', 'total', 'bad_num', 'bad_rate',
                          'cum_bad_rate', 'gain', 'ks', 'lift', 'rev_lift']]\
        .to_dict(orient='index')
