# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 23:03:35 2022

@author: Lenny
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin


class PermuteSelector(BaseEstimator, TransformerMixin):

    def __init__(self, estimator,
                 n_jobs=-1, rf_threshold=0.95):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.rf_threshold = rf_threshold

    def fit(self, X, y):
        clf = self.estimator
        clf.fit(X, y)
        ft_imp = permutation_importance(clf, X, y, n_jobs=self.n_jobs)
        # TODO
