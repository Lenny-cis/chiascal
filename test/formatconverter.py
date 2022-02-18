# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 13:50:58 2021

@author: linjianing
"""


import pandas as pd
import numpy as np
import re
import operator
import uuid
from networkx import DiGraph, draw_networkx, circular_layout
from networkx import eigenvector_centrality, has_path
from matplotlib import pyplot as plt
from .fetcher import FetchInfo

CITY_LEVEL = ['一线城市', '新一线城市', '二线城市', '三线城市',
              '四线城市', '五线城市']


def format_parse(df):
    """结构化解析."""
    def get_uuid():
        return str(uuid.uuid1()).replace("-", "").lower()

    if df.empty:
        return pd.DataFrame()
    dict_type = df.applymap(type).applymap(
        lambda x: issubclass(x, dict)).any()
    dict_type = dict_type.index[dict_type]
    if len(dict_type) > 0:
        for var in dict_type:
            parsed_data = pd.DataFrame.from_dict(df.loc[:, var].to_dict(),
                                                 orient='index')
            use_cols = df.columns.difference(parsed_data.columns)
            df = pd.concat([df.loc[:, use_cols], parsed_data], axis=1,
                           join='inner')
            df.drop(var, axis=1, inplace=True)
        df = format_parse(df)
    list_type = df.applymap(type).applymap(
        lambda x: issubclass(x, list)).any()
    list_type = list_type.index[list_type]
    if len(list_type) > 0:
        for var in list_type:
            df.loc[:, var] = df.loc[:, var].map(
                lambda x: [] if x is np.nan else x)
            parsed_data = pd.DataFrame.from_records(df.loc[:, var].to_list())
            if not parsed_data.empty:
                parsed_data = parsed_data.stack().droplevel(1)
            # df = pd.concat([df, parsed_data], axis=1, join='inner')
            df = df.merge(pd.DataFrame(parsed_data), how='inner',
                          left_index=True, right_index=True)
            df.drop(var, axis=1, inplace=True)
            df.reset_index(inplace=True, drop=True)
        df = format_parse(df)
    if '_id' not in df.columns:
        df.loc[:, '_id'] = [get_uuid() for x in range(df.shape[0])]
    else:
        df.loc[:, '_id'] = df.loc[:, '_id'].map(str)
    return df


def intconvert(ser):
    """整数转换."""
    return ser.apply(pd.to_numeric, downcast='unsigned')


def floatconvert(ser):
    """整数转换."""
    return ser.apply(pd.to_numeric, downcast='float')


def datetimeconvert(unit=None):
    """日期转换."""
    def datetimefactor(ser):
        return ser.apply(pd.to_datetime, errors='coerce', unit=unit)
    return datetimefactor


def categoryconvert(categs=None, ordered=False):
    """字典转换."""
    def _categfactor(ser):
        return ser.astype(pd.CategoricalDtype(categs, ordered))
    return _categfactor


def pctconvert(ser):
    """百分比转换."""
    return ser.str.strip('%').apply(pd.to_numeric, downcast='float',
                                    errors='coerce') / 100


def moneyconvert(ser):
    """金额转换."""
    def _money_convert(x):
        def _curr(x):
            exchange_dic = {
                '人民币': 1, '美元': 7.5, '香港元': 0.8, '香港币': 0.8,
                '港币': 0.8, '港元': 0.8, '英镑': 9, '日元': 0.06, '欧元': 8,
                '韩元': 0.006, '阿富汗尼': 0.09, '澳大利亚元': 4.99,
                '新加坡元': 5.05, '加拿大元': 5.22, '加元': 5.22,
                '德国马克': 12.01, '瑞士法郎': 7.53, '元': 1, '万': 10000,
                '亿': 100000000
                }
            for c in exchange_dic:
                if re.search(c, x) is not None:
                    x = x.replace(c, '')
                    curr, x = _curr(x)
                    curr *= exchange_dic[c]
                    return curr, x
            return 1, x

        def _num(x):
            m = re.search(r'\d+(\.)*\d*', x)
            if m is not None:
                num = re.search(r'\d+(\.)*\d*', x)[0]
                return float(num), x.replace(num, '')
            return np.nan, x

        if x == '' or x == 'null' or pd.isna(x):
            return np.nan
        if not isinstance(x, str):
            return x
        x = x.replace('(', '')
        x = x.replace(')', '')
        x = x.replace(' ', '')
        x = x.replace('null', '')
        x_l = re.split(r',|，|、', x)
        res = np.nan
        for x in x_l:
            num, x = _num(x)
            curr, x = _curr(x)
            if x != '':
                print('输入币种:{0}\n可新增至函数中'.format(x))
            res = np.nansum([res, curr * num])
        return res
    return ser.map(_money_convert)


def stockconvert(ser):
    """股权转换."""
    def _stock_convert(x):
        pattern = r'^(?P<amount>(\d+)(\.*)(\d*))(?P<unit>.*)'
        exchange_dic = {
            '元': 1, '': 10000, '股': 1, '人民币元': 1, '人民币万元': 10000,
            '万人民币元': 10000, '万元': 10000, '万股': 10000, '万': 10000,
            '万人民币': 10000, '万元人民币': 10000}
        if x == '' or pd.isna(x):
            return np.nan
        x = x.replace('(', '')
        x = x.replace(')', '')
        x = x.replace(' ', '')
        x = x.replace('null', '')
        m = re.match(pattern, x)
        if m is not None:
            try:
                exchange = exchange_dic[m.group('unit')]
                return float(m.group('amount')) * exchange
            except KeyError:
                print('期望币种:{0}\n输入币种:{1}\n可新增至函数中'.format(
                    '、'.join(exchange_dic.keys()), m.group('unit')))
        else:
            print(x)
        return np.nan
    return ser.map(_stock_convert)


def personnelsizeconvert(ser):
    """人员规模转字典."""
    def _personnelsize_to_categ(x):
        categ_list = ['小于50人', '50-99人', '100-499人', '500-999人',
                      '1000-4999人', '5000-9999人', '10000人以上']
        if x == '' or pd.isna(x):
            return np.nan
        try:
            x_n = int(x)
            idx = np.piecewise(
                x_n, [x_n < 50, 50 <= x_n < 100, 100 <= x_n < 500,
                      500 <= x_n < 1000, 1000 <= x_n < 5000,
                      5000 <= x_n < 10000],
                [0, 1, 2, 3, 4, 5, 6])
            return categ_list[idx]
        except ValueError:
            return x
    return ser.map(_personnelsize_to_categ)


def areaconvert(ser):
    """面积转换."""
    def _area_to_float(x):
        pat = r'^(?P<float>(\d+)(\.*)(\d*))(?P<tt>(万*))(?P<unit>.*)$'
        exchange_dic = {'公顷': 1,
                        '平方千米': 100,
                        '平方公里': 100,
                        '平方米': 0.0001,
                        '': 1}
        if x == '' or pd.isna(x):
            return np.nan
        x = x.replace('(', '').replace(')', '').replace(' ', '')
        m = re.match(pat, x)
        if m is not None:
            try:
                exchange = exchange_dic[m.group('unit')]
                return float(m.group('float'))\
                    * (10000 if m.group('tt') != '' else 1) * exchange
            except KeyError:
                print('期望单位:{0}\n输入单位:{1}\n可新增至函数中'.format(
                    '、'.join(exchange_dic.keys()), m.group('unit')))
        else:
            print(x)
        return np.nan
    return ser.map(_area_to_float)


def insnumconvert(ser):
    """年报保险人数转换."""
    def _keepnum(x):
        pat = r'\d+'
        if x == '' or pd.isna(x):
            return np.nan
        m = re.match(pat, x)
        if m is not None:
            return int(m[0])
        return np.nan
    return ser.map(_keepnum)


var_converters = {
    'basic': {
        'insuredSize': intconvert,
        'regStatus': categoryconvert(),
        'approvalDate': datetimeconvert(None),
        'regCapital': moneyconvert,
        'endDate': datetimeconvert(None),
        'city': categoryconvert(),
        'industry': categoryconvert(),
        'realCapital': moneyconvert,
        'personnelSize': personnelsizeconvert,
        'orgType': categoryconvert(),
        'province': categoryconvert(),
        'taxpayerQual': categoryconvert(),
        'companyType': categoryconvert(),
        'foundDate': datetimeconvert(None),
        'site': categoryconvert(),
        'regOrg': categoryconvert(),
        'startDate': datetimeconvert(None),
        'city_level': categoryconvert(categs=CITY_LEVEL, ordered=True),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'annual_report': {
        'holdCapital': moneyconvert,
        'holdDate': datetimeconvert(None),
        'realCapital': moneyconvert,
        'regDate': datetimeconvert(None),
        'realDate': datetimeconvert(None),
        'regStatus': categoryconvert(),
        'agedInsNum': insnumconvert,
        'medicalInsNum': insnumconvert,
        'birthInsNum': insnumconvert,
        'unemployInsNum': insnumconvert,
        'injuryInsNum': insnumconvert,
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'branch': {
        'regDate': datetimeconvert(None),
        'regStatus': categoryconvert(),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'abnormal': {
        'regReason': categoryconvert(),
        'regOrg': categoryconvert(),
        'regDate': datetimeconvert(None),
        'removeDate': datetimeconvert(None),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'oldabnormal': {
        'regReason': categoryconvert(),
        'regOrg': categoryconvert(),
        'regDate': datetimeconvert(None),
        'removeDate': datetimeconvert(None),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'punish': {
        'regDate': datetimeconvert(None),
        'regNo': categoryconvert(),
        'regOrg': categoryconvert(),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'change': {
        'altDate': datetimeconvert(None),
        'altItem': categoryconvert(),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'illegal': {
        'regDate': datetimeconvert(None),
        'regOrg': categoryconvert(),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'owntax': {
        'publishDate': datetimeconvert(None),
        'regDate': datetimeconvert(None),
        'newOwnTaxBalance': floatconvert,
        'ownTaxBalance': floatconvert,
        'taxAmount': floatconvert,
        'taxBalance': floatconvert,
        'location': categoryconvert(),
        'department': categoryconvert(),
        'taxCategory': categoryconvert(),
        'taxpayerType': categoryconvert(),
        'type': categoryconvert(),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'license': {
        'licenseOrg': categoryconvert(),
        'endDate': datetimeconvert(None),
        'startDate': datetimeconvert(None),
        'regDate': datetimeconvert(None),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'oldlicense': {
        'licenseOrg': categoryconvert(),
        'endDate': datetimeconvert(None),
        'startDate': datetimeconvert(None),
        'regDate': datetimeconvert(None),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'taxillegal': {'taxOrg': categoryconvert()},
    'envpunish': {
        'punish_basis': categoryconvert(),
        'punish_department': categoryconvert(),
        'publish_time': datetimeconvert(None),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'taxcredit': {
        'taxRate': categoryconvert(['A', 'B', 'M', 'C', 'D'], ordered=True),
        'type': categoryconvert(),
        'taxOrg': categoryconvert()
        },
    'check': {
        'checkResult': categoryconvert(),
        'checkOrg': categoryconvert(),
        'checkType': categoryconvert(),
        'checkDate': datetimeconvert(None),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'cert': {
        'endDate': datetimeconvert(None),
        'certType': categoryconvert(),
        'regDate': datetimeconvert(None),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'bid': {
        'regDate': datetimeconvert(None),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'patent': {
        'regDate': datetimeconvert(None),
        'type': categoryconvert(),
        'lawState': categoryconvert(),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms'),
        'pubDate': datetimeconvert(None)
        },
    'copyright': {
        'regDate': datetimeconvert(None),
        'type': categoryconvert(),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'worksright': {
        'finishDate': datetimeconvert(None),
        'regDate': datetimeconvert(None),
        'pubDate': datetimeconvert(None),
        'type': categoryconvert(),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'staff': {
        'role': categoryconvert(),
        'humanComSize': intconvert,
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'holder': {
        'role': categoryconvert(),
        'companySize': intconvert,
        'holdScale': pctconvert,
        'holdCapital': moneyconvert,
        'subConAmStr': moneyconvert,
        'scale': pctconvert,
        'joinDate': datetimeconvert(None),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'invest': {
        'regStatus': categoryconvert(),
        'regDate': datetimeconvert(None),
        'investScale': pctconvert,
        'investCapital': moneyconvert,
        'scale': pctconvert,
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'realhold': {'holdScale': pctconvert},
    'realinvest': {'investScale': pctconvert},
    'pledge': {
        'regDate': datetimeconvert(None),
        'state': categoryconvert(),
        'amount': stockconvert,
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'mortgage': {
        'amount': moneyconvert,
        'regDate': datetimeconvert(None),
        'regOrg': categoryconvert(),
        'type': categoryconvert(),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'oldmortgage': {
        'amount': moneyconvert,
        'regDate': datetimeconvert(None),
        'regOrg': categoryconvert(),
        'type': categoryconvert(),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'intelpledge': {
        'pubDate': datetimeconvert(None),
        'state': categoryconvert(),
        'type': categoryconvert(),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'landmortgage': {
        'area': areaconvert,
        'purpose': categoryconvert(),
        'create_time': datetimeconvert('ms'),
        'update_time': datetimeconvert('ms')
        },
    'judicialaid': {
        'regDate': datetimeconvert(None),
        'amount': moneyconvert,
        'type': categoryconvert(),
        'state': categoryconvert()
        },
    'oldjudicialaid': {
        'regDate': datetimeconvert(None),
        'amount': moneyconvert,
        'type': categoryconvert(),
        'state': categoryconvert()
        },
    'zhixing': {
        'regDate': datetimeconvert(None),
        'amount': moneyconvert
        },
    'oldzhixing': {
        'regDate': datetimeconvert(None),
        'amount': moneyconvert
        },
    'shixin': {
        'regDate': datetimeconvert(None),
        'caseDate': datetimeconvert(None)
        },
    'oldshixin': {
        'regDate': datetimeconvert(None),
        'caseDate': datetimeconvert(None)
        }
    }


class ConvertInfo(FetchInfo):
    """结构化解析和格式转换."""

    def __init__(self, biz, enterprises=None, seq_nos=None,
                 mylogger='print_logger'):
        super().__init__(biz, enterprises, seq_nos, mylogger)
        self.var_converts = var_converters.get(self.biz, {})

    def parse(self):
        """结构化."""
        self.records = format_parse(self.records)
        return self

    def convert(self):
        """类型转换."""
        for var, func in self.var_converts.items():
            if var not in self.records.columns:
                continue
            self.records.loc[:, var] = func(self.records.loc[:, var])
        return self

    def formatconvert(self):
        """结构化和类型转换."""
        self.parse()
        self.convert()
        return self


class RelatedInfo(FetchInfo):
    """关联体信息."""

    def __init__(self, enterprises=None, seq_nos=None,
                 mylogger='print_logger'):
        super().__init__('related', enterprises=enterprises,
                         seq_nos=seq_nos, mylogger=mylogger)

    def parse(self):
        """结构化."""
        parsed_data = pd.DataFrame.from_dict(
            self.records.loc[:, 'data'].to_dict(), orient='index')
        self.records = pd.concat([self.records, parsed_data], axis=1)
        return self

    @property
    def highly_related_corps(self):
        """高度关联企业."""
        highly_related = {}
        for comp in self.enterprises:
            affiliate_parser = Affiliates(comp)
            related_records = self.records.loc[
                self.records.loc[:, 'company_name'] == comp, :]
            if not related_records.empty:
                affiliate_parser.set_records(related_records)\
                    .extract_this_company_info().to_control_graph()
                highly_related2 = affiliate_parser\
                    .highly_related_corps(threshold=0.0)
                highly_related.update({
                    (comp, hk): {**hv, 'level': 2}
                    for hi, (hk, hv) in enumerate(highly_related2.items())})
                highly_related1 = affiliate_parser.highly_related_corps()
                highly_related.update({
                    (comp, hk): {**hv, 'level': 1}
                    for hi, (hk, hv) in enumerate(highly_related1.items())})
            else:
                highly_related.update({(comp, comp): {'level': 1}})
        highly_related = pd.DataFrame.from_dict(highly_related, orient='index')
        highly_related.index.names = ['company_name', 'related_company_name']
        highly_related.reset_index(inplace=True)
        return highly_related


class Affiliates(object):
    """亲密企业."""

    def __init__(self, company_name, affiliate_type="亲密企业", depth=3):
        self.later_than_ = pd.to_datetime("2020-07-05")\
            .tz_localize("Asia/Shanghai")
        self.now_date_ = pd.Timestamp.now()
        # 是否触发RPA调用
        self.RPA_called_ = False
        self.source_data_time_ = None
        # 关联类型
        self.type_ = affiliate_type
        # 关联深度
        self.depth_ = depth
        # 本企业名称
        self.this_company_name = company_name
        self.legal_person = None
        self.biz_code_ = [
            self.this_company_name + "_" + "related",
            self.this_company_name.replace("(", "（").replace(")", "）")
            + "_" + "related"]
        # 本企业所在关联列表
        self.related_records_ = []
        self.related_records_failed = False
        self.related_graph_ = DiGraph()
        self.this_entity_info_ = dict()
        # 本企业属性
        self.reg_status_ = None
        self.reg_capital_ = None
        self.org_type_ = None
        self.found_date_ = None
        self.industry_ = None
        self.biz_scope_ = None

    def set_records(self, records):
        """导入数据."""
        self.related_records_ = records.loc[:, "relatedComList"].iloc[0]
        self.this_entity_info_ = records.loc[:, "basicInfo"].iloc[0]
        return self

    def extract_this_company_info(self):
        """提取企业信息."""
        self.reg_status_ = self.this_entity_info_.get("regStatus")
        self.reg_capital_ = self.this_entity_info_.get("regCapital")
        self.org_type_ = self.this_entity_info_.get("companyType")
        self.found_date_ = self.this_entity_info_.get("foundDate")
        self.industry_ = self.this_entity_info_.get("industry")
        self.biz_scope_ = self.this_entity_info_.get("businessScope")
        self.legal_person = self.this_entity_info_.get("legalRepresentative")
        return self

    def to_control_graph(self):
        """生成关系图."""
        self.related_graph_ = \
            Affiliates._parse_levels_holdings_roles_dict_to_graph(
                self.related_records_, self.this_company_name,
                self.reg_status_, self.reg_capital_, self.org_type_,
                self.found_date_, self.industry_, self.biz_scope_)
        return self

    def get_related_graph(self):
        """获取关系."""
        return self.related_graph_

    @staticmethod
    def _parse_levels_holdings_roles_dict_to_graph(
            related_entity_List, this_company_name, reg_status, reg_capital,
            org_type, found_date, industry, biz_scope):
        related_graph = DiGraph()

        def percent_text_to_float(text):
            """字符串百分比转换为数字."""
            if text is None:
                return text
            if text == '':
                return 0.0
            return pd.to_numeric(
                text.replace("%", "").replace("-", "")) / 100.0

        if related_entity_List is None:
            return related_graph

        # 根部企业
        if this_company_name:
            this_company_name = this_company_name.replace("(", "（")\
                .replace(")", "）")
        related_graph.add_node("this")
        related_graph.add_node(
            this_company_name, entity_type="company", reg_status=reg_status,
            reg_capital=reg_capital, org_type=org_type, found_date=found_date,
            industry=industry, biz_scope=biz_scope)
        related_graph.add_edge("this", this_company_name, weight=0.0)

        for item in related_entity_List:
            link_type = item.get('type')
            entity_name = item.get("company_name")
            if entity_name:
                entity_name = entity_name.replace("(", "（").replace(")", "）")
            companyName = item.get("company_name")
            if companyName:
                companyName = companyName.replace("(", "（").replace(")", "）")
            hold_scale = percent_text_to_float(item.get("scale"))
            role = item.get("role")
            human_name = item.get("humanName")
            hold_capital = item.get("holdCapital")
            invest_capital = item.get("investCapital")

            link_ = item.get("link")

            # company specific
            reg_status = item.get("regStatus")
            reg_capital = item.get("regCapital")
            org_type = item.get("companyType")
            found_date = item.get("foundDate")
            industry = item.get("industry")
            biz_scope = item.get("businessScope")
            legal_person = item.get("legalRepresentative")
            root_entity = item.get("root_company")
            if root_entity:
                root_entity = root_entity.replace("(", "（").replace(")", "）")
            pre_entity = item.get("pre_name")
            if pre_entity is None:
                pre_entity = companyName
            if pre_entity:
                pre_entity = pre_entity.replace("(", "（").replace(")", "）")

            if link_type == "法人":
                assert entity_name is not None
                if entity_name in related_graph:
                    pass
                else:
                    related_graph.add_node(entity_name, entity_type="human",
                                           is_manager=False)

                if (entity_name, this_company_name) in related_graph.edges():
                    related_graph.edges[entity_name, this_company_name]['role'] = related_graph.edges[entity_name, this_company_name]['role'] + "_" + "法人" if related_graph.edges[entity_name, this_company_name].get('role') else "法人"
                else:
                    related_graph.add_edge(entity_name, this_company_name,
                                           weight=hold_scale, role="法人")

            if link_type == "法人->所有公司":
                if human_name is None:
                    human_name = pre_entity
                if human_name is None:
                    human_name = link_.split("[")[0]
                assert human_name is not None
                if human_name in related_graph:
                    pass
                else:
                    related_graph.add_node(human_name, entity_type="human",
                                           is_manager=False)

                assert entity_name is not None
                if entity_name in related_graph:
                    pass
                else:
                    related_graph.add_node(
                        entity_name, entity_type="company",
                        reg_status=reg_status, reg_capital=reg_capital,
                        org_type=org_type, found_date=found_date,
                        industry=industry, biz_scope=biz_scope,
                        legal_person=legal_person)

                if (human_name, entity_name) in related_graph.edges():
                    pass
                else:
                    related_graph.add_edge(human_name, entity_name,
                                           weight=hold_scale, role=role)

            if link_type == "董监高->所有公司":
                if human_name is None:
                    human_name = pre_entity
                if human_name is None:
                    human_name = link_.split("[")[0]
                assert human_name is not None
                if human_name in related_graph:
                    related_graph.nodes[human_name]['is_manager'] = True
                else:
                    related_graph.add_node(human_name, entity_type="human",
                                           is_manager=True)
                assert entity_name is not None
                if entity_name in related_graph:
                    pass
                else:
                    related_graph.add_node(
                        entity_name, entity_type="company",
                        reg_status=reg_status, reg_capital=reg_capital,
                        org_type=org_type, found_date=found_date,
                        industry=industry, biz_scope=biz_scope,
                        legal_person=legal_person)
                if (human_name, entity_name) in related_graph.edges():
                    pass
                else:
                    related_graph.add_edge(human_name, entity_name,
                                           weight=hold_scale, role=role)

            if link_type == "自然人股东":
                assert entity_name is not None
                if entity_name in related_graph:
                    pass
                else:
                    related_graph.add_node(entity_name, entity_type="human",
                                           is_manager=False)

                if pre_entity is None:
                    pre_entity = this_company_name

                if (entity_name, pre_entity) in related_graph.edges():
                    related_graph.edges[entity_name, pre_entity]['role'] =\
                        related_graph.edges[entity_name, pre_entity]['role'] + "_" + role if related_graph.edges[entity_name, pre_entity].get('role') else role
                    related_graph.edges[entity_name, pre_entity]['hold_capital'] =\
                        hold_capital
                    related_graph.edges[entity_name, pre_entity]['weight'] =\
                        hold_scale
                else:
                    related_graph.add_edge(
                        entity_name, pre_entity, hold_capital=hold_capital,
                        weight=hold_scale, role=role)

            if link_type == "自然人股东->所有公司":
                if human_name is None:
                    human_name = pre_entity
                if human_name is None:
                    human_name = link_.split("[")[0]
                assert human_name is not None
                if human_name in related_graph:
                    pass
                else:
                    related_graph.add_node(human_name, entity_type="human",
                                           is_manager=False)

                assert entity_name is not None
                if entity_name in related_graph:
                    pass
                else:
                    related_graph.add_node(
                        entity_name, entity_type="company",
                        reg_status=reg_status, reg_capital=reg_capital,
                        org_type=org_type, found_date=found_date,
                        industry=industry, biz_scope=biz_scope,
                        legal_person=legal_person)

                if (human_name, entity_name) in related_graph.edges():
                    pass
                else:
                    related_graph.add_edge(human_name, entity_name, weight=hold_scale, role=role)

            if link_type == "对外投资企业":
                assert entity_name is not None
                if entity_name in related_graph:
                    pass
                else:
                    related_graph.add_node(
                        entity_name, entity_type="company",
                        reg_status=reg_status, reg_capital=reg_capital,
                        org_type=org_type, found_date=found_date,
                        industry=industry, biz_scope=biz_scope,
                        legal_person=legal_person)

                if (root_entity, entity_name) in related_graph.edges():
                    pass
                else:
                    related_graph.add_edge(
                        root_entity, entity_name,
                        invest_capital=invest_capital,
                        weight=hold_scale, role="股东")

            if link_type == "企业股东":
                # assert this_company_name == pre_entity, "企业股东对应投资企业非本公司？{} : {}".format(this_company_name, pre_entity)
                assert entity_name is not None
                if pre_entity is None:
                    pre_entity = this_company_name
                related_graph.add_node(
                    entity_name, entity_type="company", reg_status=reg_status,
                    reg_capital=reg_capital, org_type=org_type,
                    found_date=found_date, industry=industry,
                    biz_scope=biz_scope, legal_person=legal_person)
                if (entity_name, pre_entity) in related_graph.edges():
                    related_graph[entity_name][pre_entity]["hold_capital"] =\
                        hold_capital
                    related_graph[entity_name][pre_entity]["weight"] =\
                        hold_scale
                    related_graph[entity_name][pre_entity]["role"] = role
                else:
                    related_graph.add_edge(
                        entity_name, pre_entity, hold_capital=hold_capital,
                        weight=hold_scale, role=role)

            if link_type == "企业股东->对外投资企业":
                if pre_entity is None:
                    pre_entity = link_.split("[")[0]
                assert pre_entity is not None and pre_entity != ""
                if pre_entity in related_graph:
                    pass
                else:
                    related_graph.add_node(pre_entity, entity_type="company")
                assert entity_name is not None
                if entity_name in related_graph:
                    pass
                else:
                    related_graph.add_node(
                        entity_name, entity_type="company",
                        reg_status=reg_status, reg_capital=reg_capital,
                        org_type=org_type, found_date=found_date,
                        industry=industry, biz_scope=biz_scope,
                        legal_person=legal_person)

                if (pre_entity, entity_name) in related_graph.edges():
                    pass
                else:
                    related_graph.add_edge(
                        pre_entity, entity_name, invest_capital=invest_capital,
                        weight=hold_scale, role="股东")

            if link_type == "企业股东->企业股东":
                if pre_entity is None:
                    pre_entity = link_.split("[")[0]
                assert pre_entity is not None and pre_entity != ""
                if pre_entity in related_graph:
                    pass
                else:
                    related_graph.add_node(pre_entity, entity_type="company")
                assert entity_name is not None
                if entity_name in related_graph:
                    pass
                else:
                    related_graph.add_node(
                        entity_name, entity_type="company",
                        reg_status=reg_status, reg_capital=reg_capital,
                        org_type=org_type, found_date=found_date,
                        industry=industry, biz_scope=biz_scope,
                        legal_person=legal_person)

                if (entity_name, pre_entity) in related_graph.edges():
                    pass
                else:
                    related_graph.add_edge(
                        entity_name, pre_entity, hold_capital=hold_capital,
                        weight=hold_scale, role="股东")

            if link_type == "企业股东->企业股东->对外投资企业":
                if pre_entity is None:
                    pre_entity = link_.split("[")[1].split(">")[1]
                assert pre_entity is not None and pre_entity != ""
                if pre_entity in related_graph:
                    pass
                else:
                    related_graph.add_node(pre_entity, entity_type="company")
                assert entity_name is not None
                if entity_name in related_graph:
                    pass
                else:
                    related_graph.add_node(
                        entity_name, entity_type="company",
                        reg_status=reg_status, reg_capital=reg_capital,
                        org_type=org_type, found_date=found_date,
                        industry=industry, biz_scope=biz_scope,
                        legal_person=legal_person)

                if (pre_entity, entity_name) in related_graph.edges():
                    pass
                else:
                    related_graph.add_edge(
                        pre_entity, entity_name, hold_capital=hold_capital,
                        weight=hold_scale, role="股东")

            if link_type == "企业股东->自然人股东":
                if pre_entity is None:
                    pre_entity = link_.split("[")[0]
                assert pre_entity is not None and pre_entity != ""
                if pre_entity in related_graph:
                    pass
                else:
                    related_graph.add_node(pre_entity, entity_type="company")
                assert entity_name is not None
                if entity_name in related_graph:
                    pass
                else:
                    related_graph.add_node(entity_name, entity_type="human",
                                           is_manager=False)

                if (entity_name, pre_entity) in related_graph.edges():
                    pass
                else:
                    related_graph.add_edge(
                        entity_name, pre_entity, hold_capital=hold_capital,
                        weight=hold_scale, role=role)

            if link_type == "企业股东->自然人股东->所有公司":
                if human_name is None:
                    human_name = pre_entity
                if human_name is None:
                    human_name = link_.split("->", 2)[1].split("[")[0]
                assert human_name is not None
                if human_name in related_graph:
                    pass
                else:
                    related_graph.add_node(human_name, entity_type="human",
                                           is_manager=False)
                assert entity_name is not None
                if entity_name in related_graph:
                    pass
                else:
                    related_graph.add_node(
                        entity_name, entity_type="company",
                        reg_status=reg_status, reg_capital=reg_capital,
                        org_type=org_type, found_date=found_date,
                        industry=industry, biz_scope=biz_scope,
                        legal_person=legal_person)

                if (human_name, entity_name) in related_graph.edges():
                    pass
                else:
                    related_graph.add_edge(human_name, entity_name,
                                           hold_capital=hold_capital,
                                           weight=hold_scale, role=role)

        return related_graph

    @staticmethod
    def draw_directed_graph(my_graph, name='out'):
        """绘制关系图."""
        draw_networkx(my_graph, pos=circular_layout(my_graph), vmin=10,
                      vmax=20, width=2, font_size=8, edge_color='black')
        picture_name = name + ".png"
        plt.savefig(picture_name)
        # print('save success: ', picture_name)
        # plt.show()

    def highly_related_corps(self, threshold=0.3):
        """高度关联."""
        highly_related_entities = dict()
        this_company_node = list(self.related_graph_.successors("this"))[0]
        highly_related_entities[this_company_node] = \
            self.related_graph_.nodes[this_company_node]
        highly_related_entities[this_company_node]["related_reason"] = "自身"

        # 本企业的大企业股东
        for (u, v, data) in self.related_graph_.in_edges(this_company_node,
                                                         data=True):
            hold_scale = data['weight']
            if hold_scale >= threshold:
                node_info = self.related_graph_.nodes[u]
                if node_info.get("entity_type") == "company":
                    highly_related_entities[u] = node_info
                    highly_related_entities[u]["related_reason"] = "本企业的大企业股东"

        # 本公司的对外控股
        for (u, v, data) in self.related_graph_.out_edges(this_company_node,
                                                          data=True):
            hold_scale = data['weight']
            if hold_scale >= threshold:
                node_info = self.related_graph_.nodes[v]
                if node_info.get("entity_type") == "company":
                    highly_related_entities[v] = node_info
                    highly_related_entities[v]["related_reason"] = "本公司的对外控股"

        # 本公司法代对外控股，或同任法代
        legal_persons = [u for (u, v, data) in self.related_graph_.in_edges(this_company_node, data=True) if data.get("role") and "法人" in data["role"]]
        if len(legal_persons) > 1:
            assert self.legal_person in legal_persons
            legal_persons = [self.legal_person]
        if len(legal_persons) == 1:
            legal_person = legal_persons[0]
            # 本法人控股
            for (u, v, data) in self.related_graph_.out_edges(legal_person,
                                                              data=True):
                hold_scale = data['weight']
                if hold_scale >= threshold:
                    node_info = self.related_graph_.nodes[v]
                    if node_info.get("entity_type") == "company":
                        highly_related_entities[v] = node_info
                        highly_related_entities[v]["related_reason"] =\
                            "本公司法代对外控股"
            # 本法人同为法人
            for (u, v, data) in self.related_graph_.out_edges(legal_person,
                                                              data=True):
                role = data['role']
                if "法人" in role:
                    node_info = self.related_graph_.nodes[v]
                    if node_info.get("entity_type") == "company":
                        highly_related_entities[v] = node_info
                        highly_related_entities[v]["related_reason"] = \
                            "本公司法代同任法代"

        # 大股东对外大持股或兼任法代
        for (u, v, data) in self.related_graph_.in_edges(this_company_node,
                                                         data=True):
            hold_scale = data['weight']
            if hold_scale >= threshold:
                for (u2, v2, data2) in self.related_graph_.out_edges(
                        u, data=True):
                    # 对外大持股或兼任法代
                    hold_scale2 = data2['weight']
                    role = data2.get("role")
                    if hold_scale2 >= threshold or (role is not None
                                                    and "法人" in role):
                        node_info2 = self.related_graph_.nodes[v2]
                        if node_info2.get("entity_type") == "company":
                            highly_related_entities[v2] = node_info2
                            highly_related_entities[v2]["related_reason"] =\
                                "大股东对外大持股或兼任法代"

        # 向上二层股东(个人或企业)的直接控股或同兼法代
        for (u1, v1, data1) in self.related_graph_.in_edges(this_company_node,
                                                            data=True):
            hold_scale1 = data1['weight']
            for (u2, v2, data2) in self.related_graph_.in_edges(u1, data=True):
                hold_scale2 = data2["weight"]
                if hold_scale1 * hold_scale2 >= threshold:
                    # 间接持有30%大企业股东
                    node_info2 = self.related_graph_.nodes[u2]
                    if node_info2.get("entity_type") == "company":
                        highly_related_entities[u2] = node_info2
                        highly_related_entities[u2]["related_reason"] =\
                            "向上二层股东间接持有30%以上的企业股东"
                    # 直接持股或同兼法代
                    for (u3, v3, data3) in self.related_graph_.out_edges(
                            u2, data=True):
                        # 对外大持股或兼任法代
                        hold_scale3 = data3['weight']
                        role3 = data3.get("role")
                        if hold_scale3 >= threshold or (role3 is not None
                                                        and "法人" in role3):
                            node_info3 = self.related_graph_.nodes[v3]
                            if node_info2.get("entity_type") == "company":
                                highly_related_entities[v3] = node_info3
                                highly_related_entities[v3]["related_reason"] = "向上二层股东(个人或企业)的直接控股或同兼法代"

        return highly_related_entities

    def loosely_related_corps(self):
        """."""
        loosely_related_entities = dict()
        undirected_graph = self.related_graph_.to_undirected()
        for node in list(self.related_graph_.nodes()):
            node_info = self.related_graph_.nodes[node]
            if "entity_type" in node_info and node_info["entity_type"] == "company":
                if has_path(undirected_graph, node, self.this_company_name):
                    loosely_related_entities[node] = node_info
        return loosely_related_entities

    def demographically_related_corps(self):
        """."""
        return

    def similarity_related_corps(self):
        """."""
        return

    def legally_related_corps(self):
        """."""
        return

    def who_is_central(self):
        """中心点."""
        sorted_centrality = eigenvector_centrality(self.related_graph_)
        # sorted_centrality = closeness_centrality(self.related_graph_)
        return sorted(sorted_centrality.items(), key=operator.itemgetter(1),
                      reverse=True)
