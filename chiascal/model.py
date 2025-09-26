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


def train_lgbclf(train_set, test_set, params):
    random_seed = 20230812
    default_cv = {
        'num_boost_round': 150,
        'early_stopping_rounds': 30,
        'verbose_eval': False
    }
    default_p = {
        'boosting_type': 'goss',
        'objective': 'binary',
        'seed': random_seed,
        'max_depth': 3,
        'learning_rate': 0.1,
        'colsample_bytree': 0.7,
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': 20
    }
    default_cv = {k: params.get(k, v) for k, v in default_cv.items()}
    params_ = {k: v for k, v in params.items()
               if k not in default_cv.keys()}
    params_ = {**default_p, **params_}
    params_['min_child_samples'] = int(params_.get('min_child_samples', 20))
    lgbm = lgb.train(
        params_, train_set, valid_sets=[train_set, test_set],
        valid_names=['trian', 'test'], **default_cv
    )
    return lgbm


def train_cv(train_set, test_set, params):
    random_seed = 20230812
    default_cv = {
        'num_boost_round': 300,
        'early_stopping_rounds': 30,
        'verbose_eval': False,
        'seed': random_seed
    }
    default_p = {
        'boosting_type': 'goss',
        'objective': 'binary',
        'seed': random_seed,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': 20
    }
    default_cv = {k: params.get(k, v) for k, v in default_cv.items()}
    params_ = {k: v for k, v in params.items()
               if k not in default_cv.keys()}
    params_ = {**default_p, **params_}
    params_['min_child_samples'] = int(params_.get('min_child_samples', 20))
#    print({k: v for k, v in params_.items() if k in ['max_depth', 'num_leaves', 'min_child_samples', 'reg_lambda', 'reg_alpha']})
    cv_eval = lgb.cv(params_, train_set, **default_cv)
    return cv_eval, {**params_, **default_cv}


def objective(params, train_set, test_set, oot_set=None, save=True):
    """目标函数."""
    def adj_auc(train_auc, test_auc):
        delt_auc_ = train_auc - test_auc
        c_abs = 1 if delt_auc_ >=0 else -1
        c_maxd = 1 if (0.02-abs(delt_auc_)) >=0 else -1
        smt_ = ((1.5+1)/2-(1.5-1)/2*c_abs)*((3+0.75)/2-(3-0.75)/2*c_maxd)
        adj_auc_ = test_auc - abs(delt_auc_)*smt_
        return adj_auc_

    start = timer()
    init_res = {'status': hpo.STATUS_OK}
    fail_res = {'status': hpo.STATUS_FAIL, 'loss': np.nan}
#    print(params)
    num_leaves = params.get('num_leaves': 31)
    max_depth = params.get('max_depth')
    min_child_samples = params.get('min_child_samples', int(train_set.num_data()/max_depth**2//4/100)*100)
    if max_depth is not None:
        if num_leaves > (2**max_depth-1):
            return fail_res
    params.update({'min_child_samples': min_child_samples})
#    lgbm = train_cv(train_set, test_set, params)
    lgbm = train_lgbclf(train_set, test_set, params)
    score_ = {(k, vk): vv for k, v in lgbm.best_score.items() for vk, vv in v.items()}
    loss = 1 - adj_auc(score_[('train', 'auc')], score_[('test', 'auc')])
    train_ks = eval_calc_ks(train_set.get_label(), lgbm.predict(train_set.get_data()))
    test_ks = eval_calc_ks(test_set.get_label(), lgbm.predict(test_set.get_data()))
    score_.update({('train', 'ks'): train_ks, ('test', 'ks'): test_ks})
    if oot_set is not None:
        oot_ks = eval_calc_ks(oot_set.get_label(), lgbm.predict(oot_set.get_data()))
        oot_auc = calc_auc(oot_set.get_label(), lgbm.predict(oot_set.get_data()))
        score_.update({('oot', 'auc'): oot_auc, ('oot', 'ks'): oot_ks})
    run_time = timeer() - start
    init_res.update({'loss': loss, 'params': {**lgbm.params, 'best_iter': lgbm.best_iteration}, 'run_time': run_time, 'metric': score_})
    return init_res

def objective_cv(params, train_set, test_set, oot_set=None, save=True):
    """目标函数."""
    def adj_auc(train_auc, test_auc):
        delt_auc_ = train_auc - test_auc
        smt_ = (1.25-0.25*np.sign(delt_auc_))*(3-2.25/2-2.25/2*np.sign(0.02-delt_auc_))
        adj_auc_ = test_auc - abs(delt_auc_)*smt_
        return adj_auc_

    start = timer()
    init_res = {'status': hpo.STATUS_OK}
    fail_res = {'status': hpo.STATUS_FAIL, 'loss': np.nan}
    num_leaves = params.get('num_leaves': 31)
    max_depth = params.get('max_depth')
    min_child_samples = params.get('min_child_samples', int(train_set.num_data()/max_depth**2//4/100)*100)
#    if max_depth is not None:
#        if num_leaves > (2**max_depth-1):
#            return fail_res
    params.update({'min_child_samples': min_child_samples})
    cv_eval, cv_params = train_cv(train_set, test_set, params)
#    print(cv_params)
    if params.get('eval_train_metric'):
        tests_metric = cv_eval['valid auc-mean'][-1] - cv_eval['valid auc-stdv'][-1]
        train_metric = cv_eval['train auc-mean'][-1] - cv_eval['train auc-stdv'][-1]
        loss = 1 - adj_auc(train_metric, tests_metric)
    else:
        loss = cv_eval['auc-stdv'][-1]/cv_eval['auc-mean'][-1]
        
    run_time = timer() - start
    init_res.update({'loss': loss, 'params': {**cv_params, 'best_iter': len(cv_eval[list(cv_eval.keys())[0]])},
                     'run_time': run_time, 'eval_res': cv_eval})
    return init_res


def eval_model(lgbm, dev_data, y_col, pdo_p=(20, 600, 0.07)):
    trn_ = {'split': 't00_Train'}
    tst_ = {'split': 't01_Test'}
    oot_ = {'split': 't02_OOT'}

    dev_nm_X = dev_data[lgbm.feature_name()]
    dev_nm_y = dev_data[y_col]
    pred_nm_dev_p = pd.Series(lgbm.predict(dev_nm_X), index=dev_nm_X.index)
    pred_nm_dev = pdo_transform(pred_nm_dev_p, *pdo_p)
    pred_nm_dev_p.name = f'cocuni_proba'
    pred_nm_dev.name = f'cocuni_score'
    dev_nm_X_pred = pd.concat([dev_nm_X, dev_nm_y, pred_nm_dev_p, pred_nm_dev],
                              join='inner', axis=1)

    gain_h_ = {}
    stress_h_ = {}
    months_mets = dev_nm_X_pred \
        .groupby('apply_month').apply(lambda g: pd.Series({
            'KS': calc_ks(g[y_col].where(lambda x: x!=0.5, 0), g[f'cocuni_score'], prob=False),
            'AUC': calc_auc(g[y_col].where(lambda x: x!=0.5, 0), g[f'cocuni_score'], prob=False)))
    for t_key, t_val in dict({'train': trn_, 'test': tst_, 'oot': oot_, 'all': {}}).items():
        pred_nm_ = multindex_filter(pred_nm_dev, dict({**t_val}))
        y_nm_ = multindex_filter(dev_nm_y, dict({**t_val}))
        y_nm_ = y_nm_.where(y_nm_!=0.5, 0)
        gain_ = gen_gaintable(y_nm_, pred_nm_, bins=20, prob=False)
        stress_ = stress_quantile(y_nm_, pred_nm_, prob=False)
        gain_h_.update({t_key: gain_})
        stress_h_.update({t_key: stress_h_})

    return dev_nm_X_pred, gain_h_, stress_h_, months_mets


def model_lgb_nm_opt(train_set, test_set, oot_set, save_file, params={}, max_evals=100, trial_early_stopping_rounds=30):
    
