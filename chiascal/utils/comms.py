# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:35:58 2021

@author: linjianing
"""

import pandas as pd
import os
# import ipykernel
# import ntpath
from decimal import Decimal, getcontext
# from notebook import notebookapp
from collections import namedtuple
os.environ['NUMEXPR_MAX_THREADS'] = '20'
# from dask.distributed import Client
# from dask import dataframe as dd
getcontext().rounding = 'ROUND_HALF_UP'

Report_tuple = namedtuple('report', 'summary detail')


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


def reduce_mem(df, **kwargs):
    """
    缩减内存.
    """
    mean_usage_b = df.memory_usage(deep=True).sum()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("memory usage before: {:03.2f} MB".format(mean_usage_mb))
    df = df.convert_dtypes()
    for key, dtp in kwargs.items():
        print(key)
        df.loc[:, key] = df.loc[:, key].astype(kwargs[key])
    ori_dtypes = df.dtypes
    uni_nums = df.nunique()
    astype_dict = {}
    for key, dtp in ori_dtypes.items():
        if key in kwargs.keys():
            continue
        uni_num = uni_nums.get(key)
        if uni_num == 0:
            continue
        if pd.api.types.is_integer_dtype(dtp):
            astype_dict.update({key: 'Int64'})
        elif pd.api.types.is_float_dtype(dtp):
            astype_dict.update({key: 'float64'})
        elif (pd.api.types.is_object_dtype(dtp)
              or pd.api.types.is_string_dtype(dtp)):
            astype_dict.update({key: pd.CategoricalDtype()})
    df = df.astype(astype_dict)
    mean_usage_b = df.memory_usage(deep=True).sum()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("memory usage after: {:03.2f} MB".format(mean_usage_mb))
    return df


def change_precision(df, precision=6):
    """."""
    def _prec(val):
        return float(Decimal(val).quantize(Decimal('0.'+'0'*precision)))

    df_dtypes = df.dtypes
    for key, dtp in df_dtypes.items():
        if pd.api.types.is_float_dtype(dtp):
            df[key] = df[key].map(_prec)
    return df


def make_x_y(df, y_name, **kwargs):
    """生成自变量和应变量."""
    return df.loc[:, df.columns.difference([y_name])],\
        df.loc[:, y_name].map(lambda x: 1 if x else 0)


def update_dict_value(orient_dict, new_dict, func):
    """根据func更新嵌套字典最内层."""
    for key, val in orient_dict.items():
        if isinstance(val, dict):
            yield from [(key, dict(update_dict_value(val, new_dict, func)))]
        elif key in new_dict.keys():
            yield (key, func(val, new_dict[key]))
        else:
            yield (key, val)


# def chunkcol_read_csv(tbl, index_cols, drop_cols=None, n_iter=5, na_values=None):
#     """分块读取csv."""
#     raw_tbl_l = []
#     if drop_cols is None:
#         drop_cols = []
#     raw_tbl = pd.read_csv(tbl, index_col=index_cols, nrows=3) \
#         .drop(drop_cols, axis=1)
#     all_cols = list(raw_tbl.columns)
#     all_cols = [col for col in all_cols if col not in drop_cols]
#     print('num columns: {}'.format(len(all_cols)))
#     num_per_iter = int(len(all_cols)/n_iter+1)
#     res_num_len(all_cols)
#     for i in range(n_iter):
#         client = Client(n_workers=20)
#         cols = all_cols[num_per_iter*i: num_per_iter*(i+1)]
#         res_num -= len(cols)
#         print('iter: {} use columns: {} rest columns: {}'
#               .format(i, len(cols), res_num))
#         raw_tbl = dd.read_csv(tbl, usecols=list(set(cols + index_cols)),
#                               na_values=na_values)
#         raw_tbl = raw_tbl.compute().set_index(index_cols)
#         client.close()
#         raw_tbl = reduce_mem(raw_tbl)
#         gc.collect()
#         raw_tbl_l.appen(raw_tbl)
#     return pd.concat(raw_tbl_l, axis=1).sort_index()


# def chunkcol_stats_read_csv(tbl, index_cols, stats, y_label, drop_cols=None, n_iter=5, na_values=None):
#     """分块读取csv并做统计性筛选."""
#     raw_tbl_l = []
#     if drop_cols is None:
#         drop_cols = []
#     raw_tbl = pd.read_csv(tbl, index_col=index_cols, nrows=3) \
#         .drop(drop_cols, axis=1)
#     all_cols = list(raw_tbl.columns)
#     all_cols = [col for col in all_cols if col not in drop_cols+[y_label]]
#     print('num columns: {}'.format(len(all_cols)))
#     num_per_iter = int(len(all_cols)/n_iter+1)
#     res_num_len(all_cols)
#     for i in range(n_iter):
#         client = Client(n_workers=20)
#         cols = all_cols[num_per_iter*i: num_per_iter*(i+1)]
#         res_num -= len(cols)
#         print('iter: {} use columns: {} rest columns: {}'
#               .format(i, len(cols), res_num))
#         raw_tbl = dd.read_csv(tbl, usecols=list(set(cols + index_cols))+[y_label],
#                               na_values=na_values)
#         raw_tbl = raw_tbl.compute().set_index(index_cols)
#         client.close()
#         raw_tbl = reduce_mem(raw_tbl)
#         gc.collect()
#         X, y = make_x_y(raw_tbl, y_label)
#         raw_tbl = stats.fit_transform(X, y)
#         raw_tbl_l.appen(raw_tbl)
#     y = pd.read_csv(
#         tbl, index_col=index_cols,
#         usecols=list(set([y_label]+index_cols)), na_values=na_values)
#     raw_tbl_l.append(y)
#     return pd.concat(raw_tbl_l, axis=1).sort_index()


def init_folder(model_version):
    """初始化项目文件夹."""
    proj_path, _ = os.path.split(os.getcwd())
    proj = os.path.split(proj_path)[1]
    dataPath = namedtuple(
        'dataPath', '''data_path raw_data_path train_path train_data_path deploy_data_path
        monitor_data_path result_path model_path final_model_path dss_path''')
    data_path = os.path.join(proj_path, 'data')
    raw_data_path = os.path.join(data_path, 'raw_data')
    train_path = os.path.join(data_path, 'train_data')
    deploy_data_path = os.path.join(data_path, 'deploy_data')
    monitor_data_path = os.path.join(data_path, 'monitor_data')
    result_path = os.path.join(proj_path, 'result')
    final_model_path = os.path.join(result_path, 'final_model')
    model_path = os.path.join(result_path, model_version)
    train_data_path = os.path.join(train_path, model_version)
    dss_path = os.path.join(r'/apps-data/jianinglin', proj)
    dp = dataPath(data_path, raw_data_path, train_path, train_data_path, deploy_data_path,
                  monitor_data_path, result_path, model_path, final_model_path,
                  dss_path)
    for k, fl in dp._asdict().items():
        if not os.path.exists(fl) and k != 'dss_path':
            os.mkdir(fl)
    return dp
