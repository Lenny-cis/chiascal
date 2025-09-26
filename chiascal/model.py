import sys
import os
import pandas as pd
import numpy as np
import math
import tqdm
import pickle
from functools import partial, reduce
from itertools import product
import re
import lightgbm as lgb
import datetime
import scipy
import re
import collections
from itertools import product
import hyperopt as hpo
from hyperopt.pyll import scope
from hyperopt.early_stop import no_progress_loss
from sklearn.metrics import roc_curve
from timeit import default_timer as timer
import argparse


from src.chiascal_webank.utils.metrics import calc_aur, calc_ks, gen_gaintable


YCOL = 'y_9m_03vs30_fillpbco60'


def pdo_transform(x, pdo, base_score, base_odss, roundn=2):
    B = -pdo/np.log(2)
    A = base_score-B*np.log(base_odds)
    if roundn is not None:
        return (A+B*np.log(x/(1-x))).round(roundn)
    return A+B*np.log(x/(1-x))


def multindex_filter(df, mi_dict={}):
    if not mi_dict:
        return df
    if len(set(list(mi_dict.keys())).difference(df.index.names))>0:
        raise ValueError('Wrong Index Name')
    idsl = df.index.nlevels * [slice(None)]
    for k, v in mi_dict.items():
        k_idx = list(df.index.names).index(k)
        if isinstance(v, (slice, list)):
            idsl[k_idx] = v
        else:
            idsl[k_idx] = [v]
    if isinstance(df, pd.DataFrame):
        return df.loc[tuple(idsl), :].copy()
    return df.loc[tuple(idsl)].copy()


def stress_quantile(y, pred, prob=False):
    tdf = pd.DataFrame({'y': y, 'pred': 1-pred if prob else pred})
    tdf.loc[:, 'cut'] = pd.qcut(
        tdf.loc[:, 'pred'], 10, labels=False, duplicates='drop')
    res = {}
    for c in sorted(tdf['cut'].unique()):
        tdf_filter = tdf.loc[tdf['cut']>=c]
        ks = calc_ks(tdf_filter['y'], tdf_filter['pred'], prob=prob)
        auc = calc_auc(tdf_filter['y'], tdf_filter['pred'], prob=prob)
        c_res = {'f{c*10}%': {'KS': ks, 'AUC': auc, 'min': tdf_filter['pred'].min()}}
        res.update(c_res)
    return res
