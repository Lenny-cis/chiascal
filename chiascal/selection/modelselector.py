# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 23:03:35 2022

@author: Lenny
"""

# import os
# import csv
import pandas as pd
import numpy as np
import math
import logging

import statsmodels.api as sm
# import lightgbm as lgb
from collections import namedtuple
# from copy import copy
# from pickle import dump, load
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif_func
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
# from scipy import stats
# from hyperopt import fmin, tpe, STATUS_OK, Trials
# from hyperopt.early_stop import no_progress_loss
# from timeit import default_timer as timer
# from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from ..utils.metrics import calc_ks, calc_auc, gen_gaintable
from ..utils import FuncRunInfo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


class TreeSelector(BaseEstimator, TransformerMixin):
    """树模型特征筛选."""

    def __init__(self, n_jobs=-1, tree_threshold=0.95):
        self.n_jobs = n_jobs
        self.tree_threshold = tree_threshold

    @FuncRunInfo(logger)
    def fit(self, X, y, n_repeats=5):
        """筛选."""
        logger.info('Start {} fit'.format(self.__class__.__name__))
        categorical_columns = [col for col, dtp in X.dtypes.items()
                               if pd.api.types.is_categorical_dtype(dtp)]
        numerical_columns = X.columns.difference(categorical_columns)
        categorical_encoder = OneHotEncoder(handle_unknown="ignore")
        numerical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean"))
            ])
        preprocessing = ColumnTransformer([
            ("cat", categorical_encoder, categorical_columns),
            ("num", numerical_pipe, numerical_columns),
            ])
        rf = Pipeline([
            ("preprocess", preprocessing),
            ("classifier", RandomForestClassifier()),
            ])
        feature_names = np.r_[categorical_columns, numerical_columns]
        pg = {'classifier__n_estimators': range(
            1, min(int(np.sqrt(len(X))), 100)),
              'classifier__max_depth': range(2, 6)}
        gsc = GridSearchCV(rf, param_grid=pg, n_jobs=self.n_jobs)
        gsc.fit(X, y)
        best_est = gsc.best_estimator_
        # raw importance
        tree_feature_imp = best_est.named_steps["classifier"]\
            .feature_importances_
        if len(categorical_columns) > 0:
            ohe = best_est.named_steps["preprocess"].named_transformers_["cat"]
            ohe_cat_idxs = np.array(
                [len(cat) for cat in ohe.categories_]).cumsum()
            ohe_feature_imps = np.split(tree_feature_imp, ohe_cat_idxs)[:-1]
            feature_imps = [np.sum(imps) for imps in ohe_feature_imps]
            feature_imps.extend(tree_feature_imp[ohe_cat_idxs[-1]:])
            feature_imps = np.array(feature_imps)
        else:
            feature_imps = np.array(tree_feature_imp)
        sorted_idx = feature_imps.argsort()[::-1]
        tree_res = dict(zip(feature_names[sorted_idx],
                            feature_imps[sorted_idx].cumsum()))
        # permute
        result = permutation_importance(best_est, X, y, n_repeats=n_repeats)
        sorted_idx = result.importances_mean.argsort()[::-1]
        per_imp = result.importances_mean[sorted_idx]
        cum_imp = per_imp.cumsum()/per_imp.sum()
        permute_res = dict(zip(feature_names[sorted_idx], cum_imp))
        res = pd.DataFrame.from_dict({'rf': tree_res, 'permute': permute_res},
                                     orient='columns')
        self.raw_importance = res
        self.tree_vars = res.loc[(res <= self.tree_threshold).any(axis=1), :]
        return self

    def transform(self, X):
        """应用筛选结果."""
        return X.loc[:, self.tree_vars.index]


class LassoLRCV(BaseEstimator, TransformerMixin):
    """lasso交叉验证逻辑回归."""

    def __init__(self, score_func=None, penalty='l1'):
        self.score_func = score_func
        self.penalty = penalty

    @FuncRunInfo(logger)
    def fit(self, X, y):
        """训练."""
        logger.info('Start {} fit'.format(self.__class__.__name__))
        X_names = X.columns.to_list()
        params = {'C': 1/np.logspace(np.log(1e-2), np.log(1), 50, base=math.e)}
        lass_lr = LogisticRegression(penalty=self.penalty, solver='liblinear')
        scorer = self.score_func
        while True:
            gscv = GridSearchCV(estimator=lass_lr, param_grid=params,
                               scoring=scorer)
            gscv.fit(X, y)
            if not any(gscv.best_estimator_.coef_.ravel() < 0):
                break
            X_names = [
                k for k, v in
                dict(zip(X_names, gscv.best_estimator_.coef_.ravel())).items()
                if v > 0]
            X = X.loc[:, X_names]
        coef_dict = dict(zip(X_names, gscv.best_estimator_.coef_.ravel()))
        self.full_model = gscv
        self.lasso_vars = [k for k, v in coef_dict.items() if v >= 0]
        self.final_model = gscv.best_estimator_
        return self

    def transform(self, X):
        """应用."""
        return pd.Series(self.final_model.predict_proba(
            X[self.lasso_vars])[:, 1], index=X.index)

    def predict(self, X):
        """预测结果."""
        return pd.Series(self.final_model.predict_proba(
            X[self.lasso_vars])[:, 1], index=X.index)

    def score(self, X, y, bins=20):
        """评估模型性能."""
        pred = self.predict(X)
        KS_val = calc_ks(y, pred)
        AUC_val = calc_auc(y, pred)
        gain_tab = gen_gaintable(y, pred, bins=bins)
        s_ = namedtuple('Score', 'KS AUC Gain_Tab')
        return s_(KS_val, AUC_val, gain_tab)


class StepwiseSelector(TransformerMixin, BaseEstimator):
    """逐步回归."""

    def __init__(self, p_value_in=0.05, p_value_out=0.01, criterion='aic',
                 value_in=0.1, value_out=0.5):
        self.p_value_in = p_value_in
        self.p_value_out = p_value_out
        self.criterion = criterion
        self.value_in = value_in
        self.value_out = value_out
        self.score_space = {}

    @FuncRunInfo(logger)
    def fit(self, X, y):
        """逐步回归."""
        logger.info('Start {} fit'.format(self.__class__.__name__))
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
                included.remove(model_exclude)

            if not changed:
                break
        self.final_model = sm.Logit(
            y, sm.add_constant(X.loc[:, included])).fit(disp=False)
        self.VIFs = {key: vif_func(X.loc[:, included].values, i)
                     for i, key in enumerate(included)}
        return self

    def predict(self, X):
        """预测结果."""
        return self.final_model.predict(
            sm.add_constant(X).loc[:, self.final_model.model.exog_names])

    def transform(self, X):
        """与预测方式相同."""
        return self.final_model.predict(
            sm.add_constant(X).loc[:, self.final_model.model.exog_names])

    def score(self, X, y, bins=20):
        """评估模型性能."""
        pred = self.predict(X)
        KS_val = calc_ks(y, pred)
        AUC_val = calc_auc(y, pred)
        gain_tab = gen_gaintable(y, pred, bins=bins)
        s_ = namedtuple('Score', 'KS AUC Gain_Tab')
        return s_(KS_val, AUC_val, gain_tab)

    def set_score(self, sample_space='Train', **kwargs):
        """记录模型评价结果."""
        ssc = ['Train', 'OOS', 'OOT']
        if sample_space not in ssc:
            raise ValueError('Sample space must in {}'.format(str(ssc)))
        score_vars = ['KS', 'AUC', 'Gain_Tab']
        inter_score_vars = set(score_vars).intersection(kwargs.keys())
        if len(inter_score_vars) <= 0:
            raise ValueError('Scores must in {}'.format(str(score_vars)))
        ss = {}
        _ = [ss.update({key: val}) for key, val in kwargs.items()
             if key in inter_score_vars]
        self.score_space.update({sample_space: ss})
        return self


class StepwiseSelector(TransformerMixin, BaseEstimator):
    """逐步回归."""

    def __init__(self, p_value_in=0.05, p_value_out=0.01, criterion='aic',
                 value_in=0.1, value_out=0.5):
        self.p_value_in = p_value_in
        self.p_value_out = p_value_out
        self.criterion = criterion
        self.value_in = value_in
        self.value_out = value_out
        self.score_space = {}

    @FuncRunInfo(logger)
    def fit(self, X, y, verbose=False):
        """逐步回归."""
        def forward(clf_0, X_const, slentry, verbose=verbose):
            nonlocal changed
            from dataclasses import dataclass, asdict
            
            @dataclass(order=True)
            class chi2_score:
                chi_square: float
                p_value: float
                name: str
            
            def ptest_score_chi_square(clf_res_0, new_col, X_const):
                from scipy.stats import chi2
                clf_1_col = list(clf_res_0.model.exog_names)+[new_col]
                clf_1 = sm.GLM(y, X_const.loc[:, clf_1_col],
                               family=sm.families.Binomial())
                p0 = clf_res_0.params
                p0.loc[new_col] = 0
                H = np.mat(clf_1.hessian(p0))
                H_i = np.linalg.inv(H)
                g = np.mat(clf_1.score(p0))
                chi_s = (-g*H_i*g.T)[0, 0]
                p = 1 - chi2.cdf(chi_s, 1)
                return chi2_score(chi_s, p, new_col)
    
            included = clf_0.exog_names
            excluded = list(set(X_const.columns)-set(included))
            clf_res_0 = clf_0.fit(disp=False)
            AEEE = [ptest_score_chi_square(clf_res_0, x, X_const) for x in excluded]
            AEEE.sort(reverse=True)
            if AEEE[0].p_value >= slentry:
                return clf_0
            
            changed = True
            included.append(AEEE[0].name)
            logger.info('Add {:30} with chiSquare {:.6}'
                        .format(AEEE[0].name, AEEE[0].chi_square))
            if verbose:
                logger.info(pd.DataFrame([asdict(x) for x in AEEE]))
            clf_1_X = X_const.loc[:, included]
            clf_1 = sm.GLM(y, clf_1_X ,family=sm.families.Binomial())
            return clf_1

        def backward(clf_0, slstay, verbose=verbose):
            nonlocal changed
            included = clf_0.exog_names
            clf_res_0 = clf_0.fit(disp=False)
            pvalues = clf_res_0.pvalues.iloc[1:].sort_values(ascending=False)
            if pvalues.iloc[0] <= slstay:
                return clf_0

            changed = True
            logger.info('Drop {:30} with pvalue {:.6}'
                        .format(pvalues.index[0], pvalues.iloc[0]))
            included.remove(pvalues.index[0])
            clf_1_X = X_const.loc[:, included]
            clf_1 = sm.GLM(y, clf_1_X ,family=sm.families.Binomial())
            return clf_1

        logger.info('Start {} fit'.format(self.__class__.__name__))
        included = ['const']
        X_const = sm.add_constant(X)
        clf_0_X = X_const.loc[:, included]
        clf_0 = sm.GLM(y, clf_0_X,family=sm.families.Binomial())
        clf_res_0 = clf_0.fit(disp=False)

        while True:
            changed = False
            # forward step
            clf_0 = forward(clf_0, X_const, self.p_value_in, verbose=verbose)
            clf_res_0 = clf_0.fit(disp=False)
            included = clf_0.exog_names
            if verbose:
                logger.info(clf_res_0.summary2())

            # backward step
            b_ = any(clf_res_0.pvalues.iloc[1:] >= self.p_value_out)
            logger.info(b_)
            while b_:
                clf_0 = backward(clf_0, self.p_value_out, verbose=verbose)
                clf_res_0 = clf_0.fit(disp=False)
                b_ = any(clf_res_0.pvalues.iloc[1:] >= self.p_value_out)

            if not changed:
                break
        self.final_model = clf_res_0
        included = [x for x in included if x!='const']
        if len(included) == 1:
            self.VIFs = {k: 1 for k in included}
            return self
        self.VIFs = {key: vif_func(X.loc[:, included].values, i)
                     for i, key in enumerate(included)}
        return self

    def predict(self, X):
        """预测结果."""
        logger.info('Start {} fit'.format(self.__class__.__name__))
        return self.final_model.predict(
            sm.add_constant(X).loc[:, self.final_model.model.exog_names])

    def transform(self, X):
        """与预测方式相同."""
        logger.info('Start {} fit'.format(self.__class__.__name__))
        return self.final_model.predict(
            sm.add_constant(X).loc[:, self.final_model.model.exog_names])

    def score(self, X, y, bins=20):
        """评估模型性能."""
        pred = self.predict(X)
        KS_val = calc_ks(y, pred)
        AUC_val = calc_auc(y, pred)
        gain_tab = gen_gaintable(y, pred, bins=bins)
        s_ = namedtuple('Score', 'KS AUC Gain_Tab')
        return s_(KS_val, AUC_val, gain_tab)

    def set_score(self, sample_space='Train', **kwargs):
        """记录模型评价结果."""
        ssc = ['Train', 'OOS', 'OOT']
        if sample_space not in ssc:
            raise ValueError('Sample space must in {}'.format(str(ssc)))
        score_vars = ['KS', 'AUC', 'Gain_Tab']
        inter_score_vars = set(score_vars).intersection(kwargs.keys())
        if len(inter_score_vars) <= 0:
            raise ValueError('Scores must in {}'.format(str(score_vars)))
        ss = {}
        _ = [ss.update({key: val}) for key, val in kwargs.items()
             if key in inter_score_vars]
        self.score_space.update({sample_space: ss})
        return self
