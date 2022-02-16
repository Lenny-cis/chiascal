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

    def __init__(self, p_value_in=0.05, p_value_out=0.01, criterion='aic',
                 value_in=0.1, value_out=0.5):
        self.p_value_in = p_value_in
        self.p_value_out = p_value_out
        self.criterion = criterion
        self.value_in = value_in
        self.value_out = value_out

    def fit(self, X, y):
        """逐步回归."""
        sign = -1 if self.criterion in ['aic', 'bic'] else 1
        included = []
        restricted_model = sm.Logit(y, pd.DataFrame(
            {'const': [1] * len(y)}, index=y.index)).fit(disp=False)
        best_f = getattr(restricted_model, self.criterion)
        while True:
            changed = False
            model_exclude = None
            model_include = None
            # forward step
            excluded = list(set(X.columns)-set(included))
            for new_column in excluded:
                model = sm.Logit(
                    y, sm.add_constant(X.loc[:, included+[new_column]]))\
                    .fit(disp=False)
                if any(model.pvalues.iloc[1:] > self.p_value_in):
                    continue
                fvalue = getattr(model, self.criterion)
                if (fvalue - best_f) * sign > self.value_in:
                    best_f = fvalue
                    model_include = new_column
                    changed = True

            if model_include is not None:
                print('Add  {:30} with {} {:.6}'
                      .format(model_include, self.criterion, best_f))
                included.append(model_include)

            if len(included) == 1:
                continue
            # backward step
            full_model = sm.Logit(
                y, sm.add_constant(X.loc[:, included])).fit(disp=False)
            best_f = getattr(full_model, self.criterion)
            for ori_column in included:
                t_col = [x for x in included if x != ori_column]
                model = sm.Logit(y, sm.add_constant(X.loc[:, t_col]))\
                    .fit(disp=False)
                if any(model.pvalues.iloc[1:] > self.p_value_out):
                    continue
                fvalue = getattr(model, self.criterion)
                if (best_f - fvalue) * sign < self.value_out:
                    best_f = fvalue
                    model_exclude = ori_column
                    changed = True
            if model_exclude is not None:
                print('Drop {:30} with {} {:.6}'
                      .format(model_exclude, self.criterion, best_f))
                included.pop(model_exclude)

            if not changed:
                break
        self.final_model = sm.Logit(
            y, sm.add_constant(X.loc[:, included])).fit(disp=False)
        return self

    def predict(self, X):
        """预测结果."""
        return self.final_model.predict(
            sm.add_constant(X).loc[:, self.final_model.model.exog_names])
