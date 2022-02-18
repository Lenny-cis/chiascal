# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:05:19 2021

@author: linjianing
"""


def groupby_func(by, func):
    def factor(df):
        col = df.columns
        var = col.difference(by)
        return df.groupby(by)[var].transform(func)
    return factor


merge_map = {
    ('license', 'oldlicense', 'cert'): 'license',
    ('punish', 'oldpunish', 'envpunish', 'abnormal', 'oldabnormal'): 'punish'
    }
dup_func_map = {
    'license': {
        (('company_name', 'licenseNo', 'regDate'), 'regDate'): groupby_func(
            ['company_name', 'licenseNo'], 'min'),
        (('company_name', 'licenseNo', 'endDate'), 'endDate'): groupby_func(
            ['company_name', 'licenseNo'], 'max')
        }
    }
duplicates_vars_map = {
    'license': {
        'sort_vars': ['company_name', 'licenseNo', 'licenseFile',
                      'regDate', 'endDate'],
        'key_vars': ['company_name', 'licenseNo']
        },
    'punish': {
        'sort_vars': ['company_name', 'regNo'],
        'key_vars': ['company_name', 'regNo']
        },
    'bid': {
        'sort_vars': ['company_name', 'project', 'regDate'],
        'key_vars': ['company_name', 'project'],
        'ascending': [True, True, False]
        },
    'patent': {
        'sort_vars': ['company_name', 'regNo', 'regDate'],
        'key_vars': ['company_name', 'regNo']
        },
    'copyright': {
        'sort_vars': ['company_name', 'regNo', 'regDate'],
        'key_vars': ['company_name', 'regNo']
        }
    }
