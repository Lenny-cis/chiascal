# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 23:03:35 2022

@author: Lenny
"""

import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
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
from scipy import stats
import logging


from ..utils.metrics import calc_ks, calc_auc, gen_gaintable

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

    def __init__(self):
        pass

    def fit(self, X, y):
        """训练."""
        logger.info('Start {} fit'.format(self.__class__.__name__))
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
        self.score_space = {}

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
                included.pop(model_exclude)

            if not changed:
                break
        self.final_model = sm.Logit(
            y, sm.add_constant(X.loc[:, included])).fit(disp=False)
        self.VIFs = {key: vif_func(X.loc[:, included], i)
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
        return {'KS': KS_val, 'AUC': AUC_val, 'Gain_Tab': gain_tab}

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
