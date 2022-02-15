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
        self.tree_vars = res_.index.to_list()

    def transform(self, X):
        """应用筛选结果."""
        return X.loc[:, self.tree_vars]


class LassoLRCV:
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


class StepwiseSelector:

    def __init__(self, criterion = 'aic', p_enter = 0.01, p_remove = 0.01,
                 p_value_enter = 0.2, intercept = False, max_iter = None,
                 return_drop = False, exclude = None):
        self.criterion = criterion
        self.p_enter = p_enter
        self.p_remove = p_remove
        self.p_value_enter = p_value_enter
        self.intercept = intercept
        self.max_iter = max_iter

    def stepwise_selection(X, y,
                           initial_list=[],
                           threshold_in=0.05,
                           threshold_out = 0.1,
                           verbose = True):
        included = list(initial_list)

        while True:
            changed=False
            # forward step
            excluded = list(set(X.columns)-set(included))
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                model = sm.Logit(
                    y, sm.add_constant(
                        pd.DataFrame(X[included+[new_column]])))\
                    .fit(disp=False)
                if any(model.pvalues > 0.05):
                    continue
                if model.df_resid > 1:
                    fvalue, fpvalue, dfdiff = model.compare_f_test(
                        restricted_model)
                    if fpvalue < threshold_in:

                ssr = model.ssr
                F = (best_ssr - ssr)/ssr/model.
                if model_aic < best_aic and model_pval < threshold_in:
                    best_aic = model_aic
                    model_include = new_column
                    changed = True
                    print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
            included.append(model_include)

            # backward step

            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included])))\
                .fit(disp=False)
            # use all coefs except intercept
            model_aic = model.aic
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max() # null if pvalues is empty
            if worst_pval > threshold_out:
                changed=True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                if verbose:
                    print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
            if not changed:
                break
        return included

    def fit(self, X, y):

def stepwise(frame, target = 'target', estimator = 'ols', direction = 'both', criterion = 'aic',
            p_enter = 0.01, p_remove = 0.01, p_value_enter = 0.2, intercept = False,
            max_iter = None, return_drop = False, exclude = None):
    """stepwise to select features
    Args:
        frame (DataFrame): dataframe that will be use to select
        target (str): target name in frame
        estimator (str): model to use for stats
        direction (str): direction of stepwise, support 'forward', 'backward' and 'both', suggest 'both'
        criterion (str): criterion to statistic model, support 'aic', 'bic'
        p_enter (float): threshold that will be used in 'forward' and 'both' to keep features
        p_remove (float): threshold that will be used in 'backward' to remove features
        intercept (bool): if have intercept
        p_value_enter (float): threshold that will be used in 'both' to remove features
        max_iter (int): maximum number of iterate
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): list of feature names that will not be dropped
    Returns:
        DataFrame: selected dataframe
        array: list of feature names that has been dropped
    """
    df, y = split_target(frame, target)

    if exclude is not None:
        df = df.drop(columns = exclude)

    drop_list = []
    remaining = df.columns.tolist()

    selected = []

    sm = StatsModel(estimator = estimator, criterion = criterion, intercept = intercept)

    order = -1 if criterion in ['aic', 'bic'] else 1

    best_score = -np.inf * order

    iter = -1
    while remaining:
        iter += 1
        if max_iter and iter > max_iter:
            break

        l = len(remaining)
        test_score = np.zeros(l)
        test_res = np.empty(l, dtype = np.object)

        if direction == 'backward':
            for i in range(l):
                test_res[i] = sm.stats(
                    df[ remaining[:i] + remaining[i+1:] ],
                    y,
                )
                test_score[i] = test_res[i]['criterion']

            curr_ix = np.argmax(test_score * order)
            curr_score = test_score[curr_ix]

            if (curr_score - best_score) * order < p_remove:
                break

            name = remaining.pop(curr_ix)
            drop_list.append(name)

            best_score = curr_score

        # forward and both
        else:
            for i in range(l):
                test_res[i] = sm.stats(
                    df[ selected + [remaining[i]] ],
                    y,
                )
                test_score[i] = test_res[i]['criterion']

            curr_ix = np.argmax(test_score * order)
            curr_score = test_score[curr_ix]

            name = remaining.pop(curr_ix)
            if (curr_score - best_score) * order < p_enter:
                drop_list.append(name)

                # early stop
                if selected:
                    drop_list += remaining
                    break

                continue

            selected.append(name)
            best_score = curr_score

            if direction == 'both':
                p_values = test_res[curr_ix]['p_value']
                drop_names = p_values[p_values > p_value_enter].index

                for name in drop_names:
                    selected.remove(name)
                    drop_list.append(name)

    r = frame.drop(columns = drop_list)

    res = (r,)
    if return_drop:
        res += (drop_list,)

    return unpack_tuple(res)
