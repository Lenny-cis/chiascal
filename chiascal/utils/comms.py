# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:35:58 2021

@author: linjianing
"""

import pandas as pd
import numpy as np
import os
import gc
import json
import urllib
import ipykernel
import ntpath
from decimal import Decimal, getcontext
from notebook import notebookapp
from collections import namedtuple
os.environ['NUMEXPR_MAX_THREADS'] = '20'
from dask.distributed import Client
from dask import dataframe as dd
getcontext().rounding = 'ROUND_HALF_UP'

Report_tuple = namedtuple('report', 'summary detail')


def reduce_mem(df, **kwargs):
	"""缩减内存."""
	mean_usage_b = df.memory_usage(deep=True).sum()
    mean_usage_mb = mean_usage_b / 1024 ** 2
	print("memory usage before: {:03.2f} MB".format(mean_usage_mb))
	df = df.convert_dtypes()
	for key, dtp in kwargs.items():
        print(key)
        df.loc[;, key] = df.loc[:, key].astype(kwargs[key])
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
        elif pd.api.types.is_float_dtype(dtyp):
            astype_dict.update({key: 'float64'})
        elif (pd.api.types.is_object_dtype(dtp)
              or pd.api.types.is_string_dtype(dtp)):
            astype_dict.update({key: pd.CategoricalDtype()})
    df = df.astype(astype_dict)
	mean_usage_b = df.memory_usage(deep=True).sum()
	mean_usage_mb = mean_usage_b / 1024 ** 2
	print("memory usage after: {:03.2f} MB".format(mean_usage_mb))
	return df

    
def make_x_y(df, y_name, **kwargs):
    """生成自变量和应变量."""
    tdf = df.convert_dtypes()
    for key, dtp in kwargs.items():
        tdf.loc[:, key] = tdf.loc[:, key].astype(kwargs[key])
    ori_dtypes = tdf.dtypes
    for key, dtp in ori_dtypes.items():
        if pd.api.types.is_integer_dtype(dtp):
            tdf.loc[:, key] = pd.to_numeric(
                tdf.loc[:, key], downcast='integer')
        if pd.api.types.is_float_dtype(dtp):
            tdf.loc[:, key] = pd.to_numeric(
                tdf.loc[:, key], downcast='float')
    tdf = tdf.convert_dtypes()
    return tdf.loc[:, tdf.columns.difference([y_name])],\
        tdf.loc[:, y_name].map(lambda x: 1 if x else 0)


def update_dict_value(orient_dict, new_dict, func):
    """根据func更新嵌套字典最内层."""
    for key, val in orient_dict.items():
        if isinstance(val, dict):
            yield from [(key, dict(update_dict_value(val, new_dict, func)))]
        elif key in new_dict.keys():
            yield (key, func(val, new_dict[key]))
        else:
            yield (key, val)
