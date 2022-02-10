# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:46:28 2022

@author: linjianing
"""
from sklearn.base import BaseEstimator, TransformerMixin

class BaseBinner(TransformerMixin, BaseEstimator):
    """探索性分箱."""

    def __init__(self, cut_cnt=50, threshold_PCT=0.025, threshold_n=None,
                 max_bin_cnt=6, I_min=3, U_min=4, cut_method='eqqt',
                 tolerance=0, n_jobs=-1):
        self.cut_cnt = cut_cnt
        self.threshold_PCT = threshold_PCT
        self.threshold_n = threshold_n
        self.max_bin_cnt = max_bin_cnt
        self.I_min = I_min
        self.U_min = U_min
        self.cut_method = cut_method
        self.tolerance = tolerance
        self.n_jobs = n_jobs
        self.bins_set = {}
