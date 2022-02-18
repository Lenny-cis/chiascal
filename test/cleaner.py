# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:03:40 2021

@author: linjianing
"""


import os
import sys
import uuid
import pandas as pd
import numpy as np
from copy import deepcopy
from ..rpa_fetcher.formatconverter import ConvertInfo, categoryconvert
from .one_table import (
    func_map, drop_vars_map, rename_vars_map)
from .multi_table import (
    duplicates_vars_map, merge_map, dup_func_map)


def annual_hold_capital():
    """年报持股金额量纲清洗."""
    def check_abnormal(level0, comp_agg):
        def factor(ser):
            year_sum = ser.sum()
            agg = comp_agg[ser.index.get_level_values(level0)[0]]
            mean_ = agg['median']
            count_ = agg['count']
            if count_ <= 2:
                return 0
            piece_val = np.piecewise(
                year_sum, [year_sum / mean_ < 1e-3, year_sum / mean_ > 1e3],
                [1, -1, 0])
            return piece_val.tolist()
        return factor

    def factor(df):
        level012 = ['company_name', 'year', 'holderName']
        level01 = ['company_name', 'year']
        level0 = 'company_name'
        capital_ser = df.set_index(level012, append=True).loc[:, 'holdCapital']
        capital_ser.dropna(inplace=True)
        capital_ser = capital_ser.loc[capital_ser != 0]
        comp_agg = capital_ser.groupby(level=level01).sum()\
            .groupby(level=level0).agg(['median', 'count'])\
            .to_dict(orient='index')
        chres = capital_ser.groupby(level=level01).transform(
            check_abnormal(level0, comp_agg))
        ser_res = np.lib.scimath.power(10000, chres) * capital_ser
        return ser_res.droplevel(level012)
    return factor


class CleanInfo:
    """单数据集的数据处理."""

    def __init__(self, funcs=func_map, drop_vars=drop_vars_map,
                 duplicates_vars=duplicates_vars_map,
                 rename_vars=rename_vars_map, merge_inst=merge_map,
                 dup_funcs=dup_func_map):
        self.fm = funcs
        self.drop_vars = drop_vars
        self.duplicates_vars = duplicates_vars
        self.rename_vars_ = rename_vars
        self.merge_inst = merge_map
        self.dup_func = dup_funcs
        self.infos = {}

    def set_infos(self, infos):
        """指定清理对象."""
        for info in infos:
            if info.records.empty:
                print('-'*10, 'No Data in {}'.format(info.biz), '-'*10)
                continue
            self.infos.update({info.biz: deepcopy(info)})
        return self

    def reg_text(self):
        """正则处理文本数据."""
        for biz, info in self.infos.items():
            tdf = info.records.copy(deep=True)
            fm = self.fm.get(biz, {})
            if fm is not None:
                for (old_var, new_var), func in fm.items():
                    old_var_ = old_var
                    if isinstance(old_var, str):
                        old_var_ = (old_var,)
                    if len(tdf.columns.intersection(old_var_)) == 0:
                        print('-'*10, '{} not in {}'.format(
                            str(old_var), biz), '-'*10)
                        tdf.loc[:, new_var] = np.nan
                    else:
                        tdf.loc[:, new_var] = func(tdf.loc[:, old_var])
            info.records = tdf
        return self

    def remove_vars(self):
        """去除多余字段."""
        for biz, info in self.infos.items():
            tdf = info.records.copy(deep=True)
            drop_vars = self.drop_vars.get('comm', [])\
                + self.drop_vars.get(biz, [])
            remove_vars = tdf.columns.intersection(drop_vars)
            tdf.drop(remove_vars, axis=1, inplace=True)
            info.records = tdf
        return self

    def rename_vars(self):
        """重命名字段."""
        for biz, info in self.infos.items():
            tdf = info.records.copy(deep=True)
            rename_vars = self.rename_vars_.get(biz, {})
            if rename_vars is None:
                continue
            tdf.rename(columns=rename_vars, inplace=True)
            info.records = tdf
        return self

    def merge_instance(self):
        """合并两个接口."""
        for orient_bizs, new_biz in self.merge_inst.items():
            intsc_bizs = set(orient_bizs).intersection(self.infos.keys())
            if len(intsc_bizs) == 0:
                continue
            if len(intsc_bizs) < len(orient_bizs):
                print('-'*10, 'No Instance: {}'.format(
                    ' '.join(set(orient_bizs).difference(intsc_bizs))), '-'*10)
            merged_df = pd.DataFrame()
            for orient_biz in intsc_bizs:
                tdf = self.infos[orient_biz].records.copy(deep=True)
                tdf.loc[:, 'api_type'] = orient_biz
                merged_df = pd.concat([merged_df, tdf], axis=0,
                                      ignore_index=True)
                if orient_biz != new_biz:
                    if new_biz not in self.infos.keys():
                        self.infos[new_biz] = self.infos[orient_biz]
                        self.infos[new_biz].biz = new_biz
                    del self.infos[orient_biz]
                del tdf
            self.infos[new_biz].records = merged_df
        return self

    def drop_duplicates(self):
        """删除重复记录."""
        for biz, info in self.infos.items():
            sort_vars = self.duplicates_vars.get(biz, {}).get('sort_vars')
            key_vars = self.duplicates_vars.get(biz, {}).get('key_vars')
            ascending = self.duplicates_vars.get(biz, {}).get('ascending')
            if sort_vars is None:
                continue
            tdf = info.records.copy(deep=True)
            tdf.sort_values(sort_vars, ascending=ascending, inplace=True)
            fm = self.dup_func.get(biz, {})
            if fm is not None:
                for (old_var, new_var), func in fm.items():
                    tdf.loc[:, new_var] = func(tdf.loc[:, old_var])
            tdf.drop_duplicates(key_vars, inplace=True)
            info.records = tdf
        return self

    def clean(self):
        """清理数据."""
        self.reg_text()
        self.rename_vars()
        self.remove_vars()
        self.merge_instance()
        self.drop_duplicates()
        return self

    def save(self, file):
        """储存到hdf文件中."""
        for biz, info in self.infos.items():
            trecords = info.records
            tdtype = trecords.dtypes
            for idx, col_type in tdtype.items():
                if col_type.name == 'category':
                    trecords.loc[:, idx] = trecords.loc[:, idx].astype(str)
            trecords.to_hdf(file, biz, mode='a', complevel=9)
