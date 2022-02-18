# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:01:17 2021

@author: linjianing
"""


import pika
import logging
import json
import uuid
import pandas as pd
from copy import deepcopy
from logging.config import dictConfig
from .queries import (
    RPA_tyc_catalog, user_app_id, MQ_URL_DEV,
    format_mongodb_and_condition, primitive_queries,
    spider_task_queue_mongo_db, tyc_hist_mongo_db, tyc_delay
    )
from .utils import format_query, concatenate_tokens
from .settings import conf


USER_APP_ID = user_app_id
dictConfig(conf)


class FetchInfo:
    """数据处理器.

    Input
        enterprises = {'enterprise_name': date}
    """

    def __init__(self, biz, enterprises=None, seq_nos=None,
                 mylogger='print_logger'):
        if enterprises is not None:
            self.enterprises = list(set([
                name.replace('(', '（').replace(')', '）')
                for name in enterprises]))
        self.seq_nos = seq_nos
        if biz not in RPA_tyc_catalog.values():
            raise ValueError('biz not in list: {0}'.format(biz))
        self.biz = biz
        self.mylogger = logging.getLogger(mylogger)
        self.rpa_state = False

    def send_rpa_task(self, nocache=False, priority=7, force_send=False,
                      sendlogger='send_logger', cache_days=15):
        """直调触发rpa采集."""
        def diff_days_now(x):
            return (pd.Timestamp.now() - pd.to_datetime(x, unit='ms')).days

        def send_mq_data(queue, resp_json):
            mq_env, mq_url = 'dev', MQ_URL_DEV
            rpa_logger.info('发送MQ%s' % [mq_env, queue])
            connection = None
            try:
                connection = pika.BlockingConnection(
                    pika.URLParameters(mq_url))
                channel = connection.channel()
                channel.queue_declare(
                    queue=queue,
                    durable=True  # 持久化，生产消费端都写
                )
                channel.basic_publish(
                    exchange='',
                    routing_key=queue,
                    body=resp_json
                )
                rpa_logger.info(
                    '发送MQ成功%s' % [mq_env, queue, resp_json[:1000] + '...']
                    )
                return True
            except Exception:
                rpa_logger.info(
                    '发送MQ失败%s' % [mq_env, queue, resp_json[:1000] + '...']
                    )
            finally:
                try:
                    if connection:
                        connection.close()
                except Exception:
                    rpa_logger.info('MQ连接关闭异常%s' % [mq_env, queue])
            return False

        def to_json(obj):
            """把class对象或python对象转换成json."""
            def convert_to_builtin_type(obj):
                """把class对象转换成dict类型的对象."""
                d = {}
                try:
                    # 非字典对象会转换异常，例如mongo对象携带的ObjectId
                    d.update(obj.__dict__)
                except Exception:
                    d = str(obj)
                return d
            try:
                # 转换特殊字符会报错
                return json.dumps(
                    obj, default=convert_to_builtin_type, ensure_ascii=False
                    )  # 显示中文
            except Exception:
                return str(obj)

        def get_uuid():
            return str(uuid.uuid1()).replace("-", "").lower()

        strstmt = 'Send RPA Task: {}'.format(self.biz)
        self.mylogger.info(strstmt)
        rpa_logger = logging.getLogger(sendlogger)
        enterprise_names = self.enterprises
        if not force_send:
            self.get_records_with_name()
            curr_records = self.records.copy(deep=True)
            if not curr_records.empty:
                ban_names = curr_records.loc[
                    curr_records.loc[:, 'update_time'].map(
                        lambda x: diff_days_now(x) < cache_days),
                    'company_name'].to_list()
                enterprise_names = list(set(enterprise_names) - set(ban_names))
        if enterprise_names is None or enterprise_names == []:
            self.rpa_state = True
            return self
        seq_list = []
        if priority <= 3:
            priority = 4
        if len(enterprise_names) <= 500:
            priority = 4
        self.mylogger.info('Send {} Task'.format(len(enterprise_names)))
        for et_name in enterprise_names:
            seq_no = '_'.join([USER_APP_ID, get_uuid()])
            body = {
                "seq_no": seq_no,
                "business": self.biz,
                "company_name": et_name,
                "no_cache": nocache,
                "priority": priority
            }
            send_mq_data("URL", to_json(body))
            seq_list.append(seq_no)
        self.seq_nos = list(set(seq_list))
        return self

    def get_records_with_name(self, db_source=spider_task_queue_mongo_db,
                              condition={}, chucksize=30000, lenlimit=True,
                              keep='last'):
        """从Mongodb中取数."""
        def _requery(enterprise_names, biz, lenlimit=lenlimit, keep=keep):
            name_indexes_dict = {
                'key_names': ['business', 'state', 'company_name'],
                'key_values': [biz, 1, {'$in': enterprise_names}]}
            find_cond = format_mongodb_and_condition(**name_indexes_dict)
            find_cond.update(condition)
            db = db_source.find(find_cond)
            result = pd.DataFrame(db)
            db.close()
            if result.empty:
                return pd.DataFrame()
            if lenlimit:
                result = result.loc[result.loc[:, 'data'].str.len() > 7, :]
            if keep is not None:
                result.sort_values([
                    'company_name', 'create_time', 'update_time'],
                    inplace=True)
                result.drop_duplicates(['company_name'], keep=keep,
                                       inplace=True)
            return result.loc[:, [
                'company_name', 'data', 'create_time', 'update_time'
                ]]

        def _requery2(enterprise_names, lenlimit=lenlimit, keep=keep):
            query_dbass_UAT_config = primitive_queries["获取申请客户信息"]
            query_dbass_UAT = format_query(
                query_dbass_UAT_config["query"],
                {"enterprise_names": concatenate_tokens(enterprise_names)})
            result = db_source.query(query_dbass_UAT)\
                .getQueryResult().copy(deep=True)
            if result.empty:
                return pd.DataFrame()
            if lenlimit:
                result = result.loc[result.loc[:, 'cert_id'].str.len() > 14, :]
            if keep is not None:
                result.sort_values([
                    'company_name', 'update_time'],
                    inplace=True)
                result.drop_duplicates(['company_name'], keep=keep,
                                       inplace=True)
            return result

        def _requery3(enterprise_names, lenlimit=lenlimit, keep=keep):
            query_dbass_UAT_config = primitive_queries["找叶企业"]
            query_dbass_UAT = format_query(
                query_dbass_UAT_config["query"],
                {"enterprise_names": concatenate_tokens(enterprise_names)})
            result = db_source.query(query_dbass_UAT)\
                .getQueryResult().copy(deep=True)
            if result.empty:
                return pd.DataFrame()
            result = result.sort_values(
                by=['root_enterprise_name', 'curr_enterprise_name',
                    'rpa_update_time'], ascending =[True, True, False])
            result.drop_duplicates(
                ['root_enterprise_name', 'curr_enterprise_name'], keep='first',
                inplace=True)
            return result

        if not hasattr(db_source, 'name'):
            db_source.name = db_source.db_
        strstmt = 'Get Data From {}: {}'.format(db_source.name, self.biz)
        self.mylogger.info(strstmt)
        enterprise_names = self.enterprises
        if enterprise_names is None:
            return self
        loops = int(len(enterprise_names)/chucksize)+1
        result = pd.DataFrame()
        for i in range(loops):
            if db_source.name == 'smedb_deal':
                t_res = _requery2(
                    enterprise_names[i*chucksize:(i+1)*chucksize],
                    lenlimit, keep)
            elif db_source.name == 'core_ent':
                t_res = _requery3(
                    enterprise_names[i*chucksize:(i+1)*chucksize],
                    lenlimit, keep)
            else:
                t_res = _requery(
                    enterprise_names[i*chucksize:(i+1)*chucksize],
                    self.biz, lenlimit, keep
                    )
            result = result.append(t_res, ignore_index=True)
            del t_res
        if not result.empty:
            result.loc[:, 'get_time'] = pd.Timestamp.now()\
                .strftime('%Y-%m-%d %H:%M:%S')
        self.records = result
        return self

    def get_seq_no(self, db_source=spider_task_queue_mongo_db,
                   condition={'state': 1}):
        """从mongodb中获取seq_no."""
        strstmt = 'Get SEQ_NO From {}: {}'.format(db_source.name, self.biz)
        self.mylogger.info(strstmt)
        cond = deepcopy(condition)
        cond.update({'business': self.biz})
        db = db_source.find(cond, {'seq_no': 1})
        result = pd.DataFrame(db)
        db.close()
        if result.shape[0] <= 0:
            self.seq_nos = None
        self.seq_nos = list(result.loc[:, 'seq_no'].unique())
        return self

    def get_records_with_seq_no(self, db_source=spider_task_queue_mongo_db,
                                condition={}, chucksize=30000):
        """按seq_no从mongodb中获取数据."""
        strstmt = 'Get Data With SEQ_NO From {}: {}'\
            .format(db_source.name, self.biz)
        self.mylogger.info(strstmt)
        seq_no = self.seq_nos
        if seq_no is None:
            return self
        loops = int(len(seq_no)/chucksize)+1
        result = pd.DataFrame()
        for i in range(loops):
            find_cond = {
                'seq_no': {'$in': seq_no[i*chucksize:(i+1)*chucksize]},
                'business': self.biz
                }
            find_cond.update(condition)
            db = db_source.find(find_cond)
            t_res = pd.DataFrame(db)
            db.close()
            result = result.append(t_res, ignore_index=True)
            del t_res
        result.loc[:, 'get_time'] = pd.Timestamp.now()\
            .strftime('%Y-%m-%d %H:%M:%S')
        self.records = result
        return self

    def check_rpa(self, db_source=spider_task_queue_mongo_db, condition={},
                  chucksize=30000):
        """检查数据库."""
        strstmt = 'Check Data With SEQ_NO From {}: {}'\
            .format(db_source.name, self.biz)
        self.mylogger.info(strstmt)
        seq_no = self.seq_nos
        if seq_no is None:
            return self
        loops = int(len(seq_no)/chucksize)+1
        result = pd.DataFrame()
        for i in range(loops):
            find_cond = {
                'seq_no': {'$in': seq_no[i*chucksize:(i+1)*chucksize]},
                'business': self.biz
                }
            find_cond.update(condition)
            db = db_source.find(find_cond, {'seq_no': 1, 'state': 1})
            t_res = pd.DataFrame(db)
            db.close()
            result = result.append(t_res, ignore_index=True)
            del t_res
        if result.empty:
            self.rpa_state = False
            return self
        if any(result.loc[:, 'state'] != 1):
            self.rpa_state = False
        else:
            self.rpa_state = True
        return self

    @property
    def unviable_seq_no(self, db_sources=[spider_task_queue_mongo_db,
                                          tyc_hist_mongo_db, tyc_delay],
                        chucksize=30000):
        """暂未返回的id."""
        if not isinstance(db_sources, list):
            db_sources = [db_sources]
        seq_no = self.seq_nos
        if seq_no is None or seq_no == []:
            return {}
        condition = {}
        res = {}
        send_error = []
        for db_source in db_sources:
            strstmt = 'Check Data With SEQ_NO From {}: {}'\
                .format(db_source.name, self.biz)
            self.mylogger.info(strstmt)
            result = pd.DataFrame()
            loops = int(len(seq_no)/chucksize)+1
            for i in range(loops):
                find_cond = {
                    'seq_no': {'$in': seq_no[i*chucksize:(i+1)*chucksize]},
                    'business': self.biz
                    }
                find_cond.update(condition)
                db = db_source.find(find_cond, {'seq_no': 1, 'state': 1})
                t_res = pd.DataFrame(db)
                db.close()
                result = result.append(t_res, ignore_index=True)
                del t_res
            if result.empty:
                if db_source.name != 'mq_source_tyc_delay':
                    send_error.append(True)
            else:
                send_error.append(False)
                db_unviable_seq = result.loc[result.loc[:, 'state'] != 1,
                                             'seq_no'].to_list()
                res.update({db_source.name: db_unviable_seq})
        if all(send_error):
            self.mylogger.info('Send Error')
            return
        return res

    def load_records_to_backup(self, db_source, chucksize=30000):
        """写入数据到备份的mongodb."""
        strstmt = 'Write Data To Backup {}: {}'.format(
            db_source.name, self.biz)
        self.mylogger.info(strstmt)
        db = db_source
        if db.name != 'TYC_RPA_History_Backup':
            print('MongoDB Error: {}'.format(db.name))
            return self
        loops = int(len(self.records)/chucksize)+1
        ll = 0
        for i in range(loops):
            _dict = self.records.iloc[i*chucksize:(i+1)*chucksize, :]\
                .to_dict(orient='index').values()
            db.insert_many(_dict)
            ll += len(_dict)
            del _dict
        strstmt = 'Insert {} Records'.format(ll)
        self.mylogger.info(strstmt)
        return self

    def load_records_to_self(self, df):
        """从外部导入数据."""
        self.records = df.copy(deep=True)
        return self
