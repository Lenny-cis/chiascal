# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 20:36:49 2021

@author: linjianing
"""


from decimal import Decimal, ROUND_HALF_UP
import math
from .settings import industry_map
import os
import sys
import numpy as np
import pandas as pd
import re
from ..rpa_fetcher.formatconverter import categoryconvert


def batch_search(patreps, othret=None):
    """匹配即替换函数工厂."""
    def searchfactory(x):
        if pd.isna(x):
            return patreps.get(np.nan) or np.nan
        for pat, rep in patreps.items():
            m = re.search(pat, x)
            if m is not None:
                return rep
        return x
    return searchfactory


def re_extract(pat, othret=None):
    """匹配即提取函数工厂."""
    def factor(x):
        if x is np.nan:
            return np.nan
        m = re.search(pat, x)
        if m is not None:
            return m.group(0)
        if othret == 'raw':
            return x
        return othret
    return factor


def re_sub(patreps):
    """正则替换."""
    def factory(x):
        if pd.isna(x):
            return np.nan
        for pat, rep in patreps.items():
            x = re.sub(pat, rep, x)
        return x
    return factory


def split(sep):
    """分割字符."""


def remove_redundancy(removefunc, formatfunc=None):
    """去除冗余同时格式转换函数工厂."""
    def redundancy_factory(ser):
        t_ser = ser.copy(deep=True)
        if issubclass(type(t_ser), pd.Series):
            t_ser = t_ser.map(removefunc)
        elif issubclass(type(t_ser), pd.DataFrame):
            t_ser = t_ser.apply(removefunc, axis=1)
        if formatfunc is None:
            return t_ser
        return formatfunc(t_ser)
    return redundancy_factory


l2_map = {('^' + x + '$'): x for x in industry_map.loc[:, 'II']}
l2_map.update({
    r'^建筑装饰、装修和其他建筑业$': '建筑装饰和其他建筑业',
    r'^装卸搬运和仓储业$': '装卸搬运和运输代理业',
    r'^多式联运和运输代理业$': '装卸搬运和运输代理业',
    r'^农、林、牧、渔专业及辅助性活动$': '农、林、牧、渔服务业',
    r'^石油、煤炭及其他燃料加工业$': '石油加工、炼焦和核燃料加工业',
    r'^开采专业及辅助性活动$': '开采辅助活动',
    r'^广播、电视、电影和录音制作业$': r'广播、电视、电影和影视录音制作业'})
l1_map = industry_map.set_index('II').loc[:, 'I'].to_dict()
l1_map.update({x: x for x in industry_map.loc[:, 'I']})
status_patrep = {
    r'^在业$': r'存续（在营、开业、在册）',
    r'^存续$': r'存续（在营、开业、在册）',
    r'^开业$': r'存续（在营、开业、在册）',
    r'^吊销，未注销$': r'吊销未注销'
    }
revoke_list = ['吊销未注销', '吊销', '注销', '撤销', '清算']
companytype_patrep = {
    r'^个体户$': '个体工商户',
    r'^个体$': '个体工商户',
    r'有限责任公司': '有限责任公司',
    r'股份有限公司': '股份有限公司',
    r'个人独资企业': '个人独资企业',
    r'股份制': '股份合作制',
    r'股份合作制': '股份合作制'
    }
altItem_patrep = {
    r'((?<!财务)负责人|(?<!\*标志的为)法定代表人|经营者|执行人)((?!电话|邮箱|信息).)*$': '法定代表人变更',
    r'高(级)*管(理)*人员|董事|经理|监事|主要(人员|成员)|组织机构(变更|备案)|标有\*标志': '高级管理人员变更',
    r'许可(经营|信息)|行业|(经营|业务)(范围|类别|方式)|(经营|主营)项目|行政审批|审批项目|行业|拟定投资总额': '经营范围变更',
    r'投资(人|者)|股东|出资|股权|外商投资|境外股东|内部股份': '股东和股权变更',
    r'注册资本|实收资本|认缴|实缴|投资总额|资金数额变更|资金数额': '注册资本变更',
    r'地址|县|住所|乡|住所|邮政编码|村|经营场所|^省$|^号$|^市$|营业场所': '企业地址变更',
    r'(?<!文件)名称|字号|简称': '企业名称变更',
    r'管辖|登记机关|工商所|所属监管|属地监管|一般代表': '登记管辖机关变更',
    r'(企业|市场主体)类(型|别)|组(织|成)形式|改制': '企业类型变更',
    r'营业起始日期|期限|执照有效期': '经营期限变更',
    r'财务负责人|联络': '其他人员变更',
    r'章程': '章程变更',
    r'.*?': '其他事项变更'
    }
othalt_patrep = {
    r'(股权转让|股东|投资(人|者)|分期实缴)(.*)(:|：)': '股东和股权变更',
    r'(主要人员(.*)(:|：))|职务|主要人员': '高级管理人员变更',
    r'(生产经营地|经营场所|住所)(.*)(:|：)': '企业地址变更',
    r'((许可(经营|信息)|(经营|一般)项目)(.*)(:|：))|销售|生产|制造|批发|零售|许可(经营|信息)|货运': '经营范围变更',
    r'企业(.*?)类型(.*)(:|：)': '企业类型变更',
    r'联络(人|员)(.*)(:|：)*': '其他人员变更',
    r'(?<!文件)名称|字号|简称(.*)(:|：)': '企业名称变更',
    r'章程': '章程变更',
    r'.*?': '其他事项变更'
    }
import_item_list = ['股东和股权变更', '高级管理人员变更', '法定代表人变更']
holdertype_patrep = {
    r'^\d.+?-\w\d.+?$': '自然人',
    r'^\d.+?$': '法人'
    }
abnormalstatus_patrep = {r'.*?': '移除', np.nan: '未移除'}
abnormaltype_patrep = {
    r'年度报告': '未按规定报送年度报告',
    r'公示.*信息((?!隐瞒|虚|假).)*$': '未按规定履行即时信息公示义务',
    r'隐瞒|虚|假': '未按国家行政法规公示真实信息',
    r'无法(.)*(取得)*联系': '公司地址异常',
    r'.*': '其他异常'
    }
punishtype_patrep = {
    r'行政拘留': '行政拘留',
    r'暂扣|吊销': '暂扣或者吊销许可证，暂扣或者吊销执照',
    r'停产|停业': '责令停产停业',
    r'违法所得|非法财物|没收': '没收违法所得，没收非法财物',
    r'罚款|处罚金|处罚*.*元|\d元|元.*处罚|罚(.*元|\d)': '罚款',
    r'警告': '警告',
    r'其他行政处罚方式': '法律、法规规定的其他行政处罚方式',
    r'.*': '警告'
    }
punish_degree_map = {
    '行政拘留': '重度', '暂扣或者吊销许可证，暂扣或者吊销执照': '重度',
    '责令停产停业': '重度', '没收违法所得，没收非法财物': '中度',
    '罚款': '中度', '警告': '轻度'}
no_patrep = {
    r'\(|（|﹝|〔|【': r'[',
    r'\)|）|﹞|〕|】': r']',
    r'\,查看其他(.\d.)条相似数据|;|\s': '',
    r'\w': lambda x: x.group(0).upper()
    }
bid_patrep = {
    r'^((\(|﹝|〔|【|\[).*(\)|﹞|〕|】|\]))*?': '',
    r'''(的|-|——|--)*(评标结果公示招标公告|((公开|预|询价|竞争性谈判|采购)*'''
    '''((中|招|流)标(（成交）)*|成交|名单|(中标|评审|评标|询价|中选)*'''
    '''结果|(中(标|选))*候选人|中选人|出让|废标|异常|更正|变更)(公(告|示))*)*?'''
    '''|合同(备案|公(告|示))*|开标记录|单一来源(采购)*公示|(评标|采购|验收)(报告|文件)*'''
    '''|(答疑)*澄清文件)''': '',
    r'\(|﹝|〔|【': r'（',
    r'\)|﹞|〕|】': r'）',
    r'(（|\()(第(.)*次.*|重新)(）|\))': '',
    r'\s|_': ''
    }
func_map = {
    'basic': {
        ('regStatus', 'regStatus'): remove_redundancy(
            batch_search(status_patrep), categoryconvert()),
        ('industry', 'industry'): remove_redundancy(
            batch_search(l2_map), categoryconvert()),
        ('companyType', 'companyType_tag'): remove_redundancy(
            re_extract(r'(?<=\(|（).+?(?=\)|）)'), categoryconvert()),
        ('companyType', 'companyType'): remove_redundancy(
            batch_search(companytype_patrep), categoryconvert()),
        ('foundDate', 'regDate'): remove_redundancy(
            lambda x: pd.to_datetime(x, unit='ms')),
        ('regStatus', 'isRevoke'): remove_redundancy(
            lambda x: x in revoke_list)
        },
    'annual_report': {
        ('holderId', 'holderType'): remove_redundancy(
            batch_search(holdertype_patrep), categoryconvert()),
        ('year', 'regDate'): remove_redundancy(
            lambda x: pd.to_datetime(str(x)+'1231'))
        },
    'change': {
        ('altItem', 'altItem'): remove_redundancy(
            batch_search(altItem_patrep), categoryconvert()),
        (('altItem', 'altAf'), 'altItem'): remove_redundancy(
            lambda x: batch_search(othalt_patrep)(x['altAf'])
            if x['altItem'] == '其他事项变更'
            else x['altItem'], categoryconvert()),
        ('altItem', 'importantItem'): remove_redundancy(
            lambda x: x in import_item_list)
        },
    'holder': {
        ('holderId', 'holderType'): remove_redundancy(
            batch_search(holdertype_patrep), categoryconvert()),
        ('create_time', 'regDate'): remove_redundancy(
            lambda x: pd.to_datetime(x, unit='ms'))
        },
    'abnormal': {
        ('removeReason', 'abnormal_status'): remove_redundancy(
            batch_search(abnormalstatus_patrep, '未移除'), categoryconvert()),
        ('regReason', 'abnormal_type'): remove_redundancy(
            batch_search(abnormaltype_patrep), categoryconvert()),
        ('regReason', 'punish_type'): remove_redundancy(lambda x: '警告'),
        ('punish_type', 'punish_degree'): remove_redundancy(punish_degree_map)
        },
    'oldabnormal': {
        ('removeReason', 'abnormal_status'): remove_redundancy(
            batch_search(abnormalstatus_patrep, '未移除'), categoryconvert()),
        ('regReason', 'abnormal_type'): remove_redundancy(
            batch_search(abnormaltype_patrep), categoryconvert()),
        ('regReason', 'punish_type'): remove_redundancy(lambda x: '警告'),
        ('punish_type', 'punish_degree'): remove_redundancy(punish_degree_map)
        },
    'owntax': {
        ('company_name', 'punish_type'): remove_redundancy(lambda x: '罚款'),
        ('punish_type', 'punish_degree'): remove_redundancy(punish_degree_map)
        },
    'oldowntax': {
        ('company_name', 'punish_type'): remove_redundancy(lambda x: '罚款'),
        ('punish_type', 'punish_degree'): remove_redundancy(punish_degree_map)
        },
    'punish': {
        ('result', 'punish_type'): remove_redundancy(
            batch_search(punishtype_patrep), categoryconvert()),
        ('punish_type', 'punish_degree'): remove_redundancy(punish_degree_map)
        },
    'oldpunish': {
        ('result', 'punish_type'): remove_redundancy(
            batch_search(punishtype_patrep), categoryconvert()),
        ('punish_type', 'punish_degree'): remove_redundancy(punish_degree_map)
        },
    'license': {
        ('licenseNo', 'licenseNo'): remove_redundancy(
            re_sub(no_patrep))
        },
    'oldlicense': {
        ('licenseNo', 'licenseNo'): remove_redundancy(
            re_sub(no_patrep))
        },
    'envpunish': {
        ('punish_content', 'punish_type'): remove_redundancy(
            batch_search(punishtype_patrep), categoryconvert()),
        ('punish_number', 'punish_number'): remove_redundancy(
            re_sub(no_patrep)),
        ('punish_type', 'punish_degree'): remove_redundancy(punish_degree_map)
        },
    'cert': {
        ('certNo', 'certNo'): remove_redundancy(re_sub(no_patrep))
        },
    'bid': {
        ('title', 'project'): remove_redundancy(re_sub(bid_patrep))
        },
    'mortgage': {
        ('regNo', 'regNo'): remove_redundancy(re_sub(no_patrep))
        },
    'oldmortgage': {
        ('regNo', 'regNo'): remove_redundancy(re_sub(no_patrep))
        },
    'intelpledge': {
        ('regNo', 'regNo'): remove_redundancy(re_sub(no_patrep))
        },
    'landmortgage': {
        ('beginEndDate', 'regDate'): remove_redundancy(
            lambda x: pd.to_datetime(x.split('至')[0])),
        ('beginEndDate', 'endDate'): remove_redundancy(
            lambda x: pd.to_datetime(x.split('至')[1]))
        }
    }
drop_vars_map = {
    'comm': ['web_url', 'companyUrl', 'companyId', 'tag_str', 'cache_ymd',
             'cache_time', 'cost', 'allSize', 'is_size_ok', 'totalSize',
             'web_source', 'latest_name', 'site', 'legalRepresentativeUrl',
             'holderUrl', 'companyName', 'create_time', 'update_time',
             'get_time', 'pre_name', 'pre_id'],
    'landmortgage': ['beginEndDate'],
    'staff': ['staffUrl']
    }
rename_vars_map = {
    'patent': {'regDate': 'submitDate', 'pubDate': 'regDate',
               'patentName': 'intelName', 'patentNo': 'regNo'},
    'copyright': {'rightName': 'intelName'},
    'cert': {'certType': 'licenseFile', 'certNo': 'licenseNo'},
    'license': {'startDate': 'regDate'},
    'oldlicense': {'startDate': 'regDate'},
    'envpunish': {'punish_number': 'regNo', 'punish_content': 'result',
                  'punish_department': 'regOrg', 'punish_reason': 'content',
                  'publish_time': 'regDate'},
    'abnormal': {'regReason': 'content', 'abnormal_status': 'result'},
    'oldabnormal': {'regReason': 'content', 'abnormal_status': 'result'},
    'change': {'altDate': 'regDate'},
    'owntax': {'publishDate': 'regDate'},
    'oldowntax': {'publishDate': 'regDate'}
    }
