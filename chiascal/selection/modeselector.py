# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 23:03:35 2022

@author: Lenny
"""

import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from scipy import stats


class TreeSelector(BaseEstimator, TransformerMixin):
    """树模型特征筛选."""

    def __init__(self, estimator,
                 n_jobs=-1, tree_threshold=0.95, permute=False):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.tree_threshold = tree_threshold
        self.permute = permute

    def fit(self, X, y):
        """筛选."""
        imp_mean = SimpleImputer(missing_values=np.nan,
                                 strategy='most_frequent')
        X = imp_mean.fit_transform(X)
        clf = self.estimator
        clf.fit(X, y)
        if self.permute:
            ft_imp = permutation_importance(
                clf, X, y, n_jobs=self.n_jobs).importances_mean
        ft_imp_df = pd.Series(ft_imp, index=X.columns).sort_values(
            ascending=False)
        cm_ = ft_imp_df.cumsum()/ft_imp_df.sum()
        res_ = cm_.loc[cm_ < self.tree_threshold]
        self.tree_vars = res_

    def transform(self, X):
        """应用筛选结果."""
        return X.loc[:, self.tree_vars.index]


class LassoLRCV(BaseEstimator, TransformerMixin):
    """lasso交叉验证逻辑回归."""

    def __init__(self):
        pass

    def fit(self, X, y):
        """训练."""
        X_names = X.columns.to_list()
        params = {'C': 1/np.logspace(np.log(1e-6), np.log(1), 50, base=math.e)}
        lass_lr = LogisticRegression(penalty='l1', solver='liblinear')
        while True:
            gscv = GridSearchCV(lass_lr, params)
            gscv.fit(X, y)
            if not any(gscv.best_estimator_.coef_.ravel() < 0):
                break
            X_names = [
                k for k, v in
                dict(zip(X_names, gscv.best_estimator_.coef_.ravel())).items()
                if v > 0]
            X = X.loc[:, X_names]
        coef_dict = dict(zip(X_names, gscv.best_estimator_.coef_.ravel()))
        self.lasso_vars = [k for k, v in coef_dict.items() if v > 0]
        return self

    def transform(self, X):
        """应用."""
        return X.loc[:, self.lasso_vars]


class StepwiseSelector(BaseEstimator, TransformerMixin):
    """逐步回归."""

    def __init__(self, threshold_in=0.05, threshold_out=0.1):
        self.threshold_in = threshold_in
        self.threshold_out = threshold_out

    def fit(self, X, y):
        """逐步回归."""
        included = []
        restricted_model = sm.Logit(y, pd.DataFrame({'const': [1] * len(y)}))\
            .fit(disp=False)
        best_f = 0
        while True:
            changed = False
            model_exclude = None
            model_include = None
            # forward step
            excluded = list(set(X.columns)-set(included))
            for new_column in excluded:
                model = sm.Logit(
                    y, sm.add_constant(
                        pd.DataFrame(X[included+[new_column]])))\
                    .fit(disp=False)
                if any(model.pvalues.iloc[1:] > 0.05):
                    continue
                fvalue, fpvalue, _ = model.compare_f_test(restricted_model)
                if fpvalue < self.threshold_in and fvalue > best_f:
                    best_f = fvalue
                    model_include = new_column
                    changed = True

            if model_include is not None:
                print('Add  {:30} with p-value {:.6}'
                      .format(model_include, fpvalue))
                included.append(model_include)

            if len(included) == 1:
                continue
            # backward step
            full_model = sm.Logit(
                y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=False)
            best_f = np.inf
            for ori_column in included:
                t_col = [x for x in included if x != ori_column]
                model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[t_col])))\
                    .fit(disp=False)
                if any(model.pvlaues.iloc[1:] > 0.05):
                    continue
                fvalue, fpvalue, _ = full_model.compare_f_test(model)
                if fpvalue > self.threshold_out and fvalue < best_f:
                    best_f = fvalue
                    model_exclude = ori_column
                    changed = True
            if model_exclude is not None:
                print('Drop {:30} with p-value {:.6}'
                      .format(model_exclude, fpvalue))
                included.pop(model_exclude)

            if not changed:
                break
        self.stepwise_vars = included
        return self

    def transform(self, X):
        """应用."""
        return X.loc[:, self.stepwise_vars]
