# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:35:58 2021

@author: linjianing
"""

import pandas as pd


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
