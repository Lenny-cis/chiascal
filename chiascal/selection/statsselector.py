# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:42:52 2022

@author: linjianing
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
from types import MethodType
from collections import Counter, OrderedDict
from scipy import stats
import logging


from ..utils.metrics import calc_iv
from ..utils.cut_merge import gen_cut, gen_cross
from ..utils.progress_bar import make_tqdm_iterator
from ..utils import FuncRunInfo, Register, Report_tuple


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
    if ser.isna().all():
        return np.nan
    max_val = stats.mode(ser.dropna(), nan_policy='omit', keepdims=False)[1]
    return max_val / ser.shape[0]


def num_unique(ser):
    """不同值的个数."""
    return ser.nunique()


def iv(ser, y):
    """变量IV."""
    if all(ser.isna()):
        return np.nan
    cut = gen_cut(ser, n=20, mthd='eqqt', precision=4)
    cross, cut = gen_cross(ser, y, cut)
    iv_ = calc_iv(cross)
    return iv_


def psi(base, rept):
    """变量PSI."""
    cut = gen_cut(base, n=20, method='eqqt', precision=6)
    ser = pd.concat([base, rept], axis=0, keys=[0, 1])
    y = pd.Series(ser.index.get_level_values(0), index=ser.index)
    cross, cut = gen_cross(ser, y, cut)
    return calc_iv(cross)


class StatsSelector(TransformerMixin, BaseEstimator):
    """分箱前变量筛选."""
    register_funcs = Register()

    def __init__(self, nomissing=0.05, noconcentration=0.05, nunique=0,
                 IV=0.01, n_jobs=-1):
        self.thresholds = {
            'nomissing': nomissing,
            'noconcentration': noconcentration,
            'nunique': nunique,
            'IV': IV}
        self.n_jobs = n_jobs
        self.input_vars = {}
        self.output_vars = {}

    def bind_funcs(self, funcs):
        """绑定外部函数."""
        if not isinstance(funcs, list):
            funcs = [funcs]
        for func in funcs:
            setattr(self, func.__name__, MethodType(func, self))
            self.register_funcs[func.__name__] = func
        return self

    def bind_funcs_thresholds(self, ft_dict):
        """绑定外部函数及阈值."""
        self.bind_funcs(list(ft_dict.keys()))
        _ = [self.thresholds.update({key.__name__: val})
             for key, val in ft_dict.items()]
        return self

    @register_funcs.register
    def nomissing(self, ser, y=None):
        """."""
        return 1 - missing_ratio(ser)

    @register_funcs.register
    def noconcentration(self, ser, y=None):
        """."""
        return 1 - concentration_ratio(ser)

    @register_funcs.register
    def nunique(self, ser, y=None):
        """."""
        return num_unique(ser)

    @register_funcs.register
    def IV(self, ser, y=None):
        """."""
        return iv(ser, y)

    def _var_fit(self, x, y):
        res_ = {
            key: val(self, x, y)
            for key, val in self.register_funcs.items()
        }
        res_.update({'drop_reason': key for key, val in res_.items()
                     if val < self.thresholds.get(key)})
        return res_

    @FuncRunInfo(logger)
    def fit(self, X, y, refit=True, **kwargs):
        """筛选."""
        if not refit:
            return self
        tqdm_options = {'iterable': X.columns.tolist(), 'disable': False,
                        'desc': 'Stats'}
        progress_bar = make_tqdm_iterator(**tqdm_options)

        # stats = []
        # for x_name in X.columns:
        #     stats.append(self._var_fit(X.loc[:, x_name], y))
        stats = Parallel(n_jobs=self.n_jobs)(
            delayed(self._var_fit)(X[x_name], y)
            for x_name in progress_bar)
        tdict = dict(zip(X.columns.tolist(), stats))
        self.input_vars.update(tdict)
        self.output_vars.update({
            key: val for key, val in tdict.items()
            if pd.isna(val.get('drop_reason'))})
        return self

    @FuncRunInfo(logger)
    def transform(self, X):
        """应用."""
        tran_x = [x_n for x_n in X.columns if x_n in self.output_vars.keys()]
        return X.loc[:, tran_x]

    def get_IVs(self):
        """."""
        return {key: val['IV'] for key, val in self.output_vars.items()}

    @property
    def rept(self):
        """Report DF."""
        fil = OrderedDict({k: 0 for k in self.register_funcs.keys()})
        fil.update({
            key: val for key, val in Counter([
                r.get('drop_reason')
                for r in self.input_vars.values()]).items()
            if key is not None})
        output_ = OrderedDict(zip(
            fil.keys(),
            len(self.input_vars.keys()) - np.cumsum(list(fil.values()))))
        input_ = OrderedDict({k: len(self.input_vars.keys())
                              for k in self.register_funcs.keys()})
        input_.update(dict(zip(
            list(input_.keys())[1:], list(output_.values())[:-1])))
        summ = pd.DataFrame({
            'input': input_, 'filter': fil, 'output': output_})
        det = pd.DataFrame.from_dict(self.input_vars, orient='index')
        return Report_tuple(summ, det)


class PSISelector(TransformerMixin, BaseEstimator):
    """PSI筛选."""
    def __init__(self, psi=0.1, n_jobs=-1):
        self.psi = psi
        self.n_jobs = n_jobs
        self.input_vars = {}
        self.output_vars = {}

    @FuncRunInfo(logger)
    def fit(self, X, y, rep_X):
        """筛选."""
        tqdm_options = {'iterable': X.columns.tolist(), 'disable': False,
                        'desc': 'Stats'}
        progress_bar = make_tqdm_iterator(**tqdm_options)
        psi_fil = Parallel(n_jobs=self.n_jobs)(
            delayed(psi)(X[x_name], rep_X[x_name])
            for x_name in progress_bar)
        tdict = dict(zip(X.columns.tolist(), psi_fil))
        self.input_vars.update(tdict)
        self.output_vars.update({
            key: val for key, val in tdict.items()
            if val <= self.psi})
        return self

    def transform(self, X):
        """应用."""
        tran_x = [x_n for x_n in X.columns if x_n in self.output_vars.keys()]
        return X.loc[:, tran_x]

    @property
    def rept(self):
        """Report DF."""
        in_num = len(self.input_vars.keys())
        ou_num = len(self.output_vars.keys())
        det = pd.concat([pd.Series(self.input_vars, name='psi'),
                         pd.Series({key: 1 for key in self.output_vars},
                                   name='output_flag')], axis=1)
        summ = pd.DataFrame.from_dict({
            'PSI': {'input': in_num, 'filter': in_num-ou_num,
                    'output': ou_num}},
            orient='index')
        return {'summary': summ, 'detail': det}

# def var_stats(ser, y, thresholds):
#     """变量统计信息."""
#     missing_ratio_ = missing_ratio(ser)
#     concentration_ratio_ = concentration_ratio(ser)
#     num_unique_ = num_unique(ser)
#     iv_ = iv(ser, y)
#     res_ = {'nomissing': 1 - missing_ratio_,
#             'noconcentration': 1-concentration_ratio_,
#             'nunique': num_unique_,
#             'IV': iv_}
#     res_.update({'drop_reason': key for key, val in res_.items()
#                  if val < thresholds.get(key)})
#     return res_
