"""
数据源：处理各类数据连接，数据异常处理，以及初步ETL， 数据格式统一化等操作

"""

from impala.dbapi import connect
from impala.util import as_pandas
import pandas as pd
import numpy as np
import pymysql
import pymongo
import json
from neo4j import GraphDatabase
from DBUtils.PooledDB import PooledDB
from happybase import ConnectionPool
# from datasketch.experimental.aio.lsh import AsyncMinHashLSH
import asyncio


def bytes_structure_to_string(data):
    if isinstance(data, bytes):
        return data.decode()
    if isinstance(data, (str, int)):
        return str(data)
    if isinstance(data, dict):
        return dict(map(bytes_structure_to_string, data.items()))
    if isinstance(data, tuple):
        return tuple(map(bytes_structure_to_string, data))
    if isinstance(data, list):
        return list(map(bytes_structure_to_string, data))
    if isinstance(data, set):
        return set(map(bytes_structure_to_string, data))


def min_hash_enterprise_name(enterprise_name, administrative_divisions_pattern, organization_pattern,
                             stop_words_pattern, num_perm=256):
    vector = parse_and_quantize(enterprise_name, administrative_divisions_pattern, organization_pattern,
                                stop_words_pattern, father_end_signal_patt)
    print("vectorized enterprise_name :{}".format(vector))
    m1 = MinHash(num_perm=num_perm)
    for d in vector:
        m1.update(d.encode('utf8'))
    return m1


class DBconnector(object):
    def __init__(self, DBType, host=None, port=None, db=None, user=None, password=None, url=None,
                 pool=True, max_connections=8):
        self.pool_ = pool
        self.max_connections_ = max_connections
        self.DBType_ = DBType
        self.host_ = host
        self.port_ = port
        self.user_ = user
        self.password_ = password
        self.db_ = db
        self.url_ = url
        self.cursor_ = self.connect()
        self.queryResult_ = None

    def connect(self):
        if self.DBType_ == 'impala':
            conn = connect(host=self.host_, port=self.port_, user=self.user_, password=self.password_)
            return conn.cursor(user=self.user_)
        if self.DBType_ == 'mysql':
            if not self.pool_:
                connection = pymysql.connect(host=self.host_, port=self.port_, db=self.db_, user=self.user_,
                                             password=self.password_, charset='utf8', autocommit=True)
                return connection
            else:
                pool = PooledDB(creator=pymysql, maxconnections=self.max_connections_,
                                mincached=2, maxcached=6, maxshared=2, blocking=True,
                                maxusage=None, setsession=['SET AUTOCOMMIT = 1'], ping=0,
                                host=self.host_, port=self.port_,
                                user=self.user_, password=self.password_,
                                database=self.db_, charset='utf8'
                                )
                connection = pool.connection()
                return connection
        if self.DBType_ == 'hive':
            conn = connect(host=self.host_, port=self.port_, user=self.user_, password=self.password_, auth_mechanism='PLAIN')
            return conn.cursor(user=self.user_)

    def query(self, query):
        if self.DBType_ == 'impala' or self.DBType_ == 'hive' or self.DBType_ == 'phoenix':
            self.cursor_.execute(query)
        if self.DBType_ == 'mysql':
            self.cursor_.ping(reconnect=True)
            self.queryResult_ = pd.read_sql(query, self.cursor_)

        return self

    def getQueryResult(self):
        return self.queryResult_

    def as_pandas(self):
        if self.DBType_ == 'impala' or self.DBType_ == 'hive' or self.DBType_ == 'phoenix':
            self.queryResult_ = as_pandas(self.cursor_)
            self.queryResult_.replace(to_replace=np.nan, value='', inplace=True)
        if self.DBType_ == 'mysql':
            self.queryResult_.replace(to_replace=np.nan, value='', inplace=True)
        return self

    def typeConverter(self, convertersDict):
        columns = self.queryResult_.columns.tolist()
        for column in convertersDict:
            if column in columns:
                if convertersDict[column] == 'datetime':
                    self.queryResult_[column] = pd.to_datetime(self.queryResult_[column], utc=True,
                                                               infer_datetime_format=True).dt.tz_convert('Asia/Shanghai')
                if convertersDict[column] == 'string':
                    self.queryResult_[column] = self.queryResult_[column].astype(str)
                if convertersDict[column] == 'numeric':
                    self.queryResult_[column] = pd.to_numeric(self.queryResult_[column])
                if convertersDict[column] == 'time':
                    self.queryResult_[column] = pd.to_datetime(self.queryResult_[column], infer_datetime_format=True).\
                        dt.tz_localize('Asia/Shanghai')
                if convertersDict[column] == 'time_no_tz':
                    self.queryResult_[column] = pd.to_datetime(self.queryResult_[column], infer_datetime_format=True)
                if convertersDict[column] == 'datetime1':
                    self.queryResult_[column] = pd.to_datetime(self.queryResult_[column], utc=True, unit='ms',
                                                               infer_datetime_format=True).dt.tz_convert('Asia/Shanghai')
        return self

    def setIndex(self, target_column):
        self.queryResult_ = self.queryResult_.set_index(target_column)
        return self

    def write_to_mysql(self, query, symbol_value_dict):
        # query = query.format(**symbol_value_dict)
        # self.query(query)
        self.cursor_.ping(reconnect=True)
        self.cursor_.cursor().execute(query, tuple(symbol_value_dict.values()))
        self.cursor_.commit()

    def close(self):
        self.cursor_.close()


class MongoDB(object):

    def __init__(self, uri, user=None, password=None, database=None):
        if database is not None:
            self._client = pymongo.MongoClient('mongodb://{}/'.format(uri), maxPoolSize=20)[database]
        else:
            self._client = pymongo.MongoClient('mongodb://{}/'.format(uri), maxPoolSize=20)
        if user is not None:
            self._client.authenticate(user, password)

    def close(self):
        self._client.close()

    def get_or_switch_to(self, key):
        return self._client[key]

    def write_dict_to_mongo(self, dict_, collection_name, doc_name):
        mongo_db_database = self.get_or_switch_to(collection_name)
        to_insert_document = mongo_db_database[doc_name]
        _id = to_insert_document.insert(dict_, check_keys=False)
        print("写入MongoDB成功！")
        return _id

    def remove_item_in_mongo(self, object_id_, collection_name, doc_name):
        mongo_db_database = self.get_or_switch_to(collection_name)
        to_insert_document = mongo_db_database[doc_name]
        to_insert_document.remove({"_id": object_id_})
        print("删除 {} from MongoDB成功！".format(object_id_))

    @staticmethod
    def write_dataframe_to_arctic_mongo(token, data_frame, arctic_store_instance, library_name, meta_data_dict):
        library = arctic_store_instance[library_name]
        library.write(token, data_frame, meta_data_dict)
        print("写入dataframe至mongo数据库成功：{}".format(token))

    @staticmethod
    def get_data_from_arctic(token, arctic_store_instance, library_name):
        library = arctic_store_instance[library_name]
        return library.read(token).data

    @staticmethod
    def has_symbol(token, arctic_store_instance, library_name):
        return arctic_store_instance[library_name].has_symbol(token)

    @staticmethod
    def _create_and_return_greeting(tx, message):
        result = tx.run("CREATE (a:Greeting) "
                        "SET a.message = $message "
                        "RETURN a.message + ', from node ' + id(a)", message=message)
        return result.single()[0]

    @staticmethod
    def df_to_mongo(self, data_frame, db_name, doc_name):
        def df2bson(df):
            """DataFrame类型转化为Bson类型"""
            data_ = json.loads(df.T.to_json()).values()
            return data_

        mongo_db = self._client[db_name]
        bson_data = df2bson(data_frame)
        to_insert_doc = mongo_db[doc_name]
        result = to_insert_doc.insert_many(bson_data)
        return result


class Neo4jDB(object):

    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def print_greeting(self, message):
        with self._driver.session() as session:
            greeting = session.write_transaction(self._create_and_return_greeting, message)
            print(greeting)

    @staticmethod
    def add_friend(tx, name, friend_name):
        tx.run("MERGE (a:Person {name: $name}) "
               "MERGE (a)-[:KNOWS]->(friend:Person {name: $friend_name})",
               name=name, friend_name=friend_name)

    @staticmethod
    def print_friends(tx, name):
        for record in tx.run("MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name = $name "
                             "RETURN friend.name ORDER BY friend.name", name=name):
            print(record["friend.name"])

    # with driver.session() as session:
    #     session.write_transaction(add_friend, "Arthur", "Guinevere")
    #     session.write_transaction(add_friend, "Arthur", "Lancelot")
    #     session.write_transaction(add_friend, "Arthur", "Merlin")
    #     session.read_transaction(print_friends, "Arthur")

    @staticmethod
    def _create_and_return_greeting(tx, message):
        result = tx.run("CREATE (a:Greeting) "
                        "SET a.message = $message "
                        "RETURN a.message + ', from node ' + id(a)", message=message)
        return result.single()[0]


class HBaseDB(object):
    def __init__(self, host, port, pool_size=4, table=None):
        self.pool = ConnectionPool(size=pool_size, host=host, port=port)
        self.table_ = table
        self.result_ = None

    def query_rows(self, row_indexes):

        if not isinstance(row_indexes, list):
            row_indexes = [row_indexes]

        with self.pool.connection() as conn:
            table = conn.table(self.table_)
            self.result_ = table.rows(row_indexes)
            conn.close()

        return self

    def get_result(self, convert_bytes=True):
        if convert_bytes:
            return bytes_structure_to_string(self.result_)
        else:
            return self.result_

    def switch_to_table(self, table_name):
        self.table_ = table_name
        return self


# class LshDB(object):

#     def __init__(self, num_perm, threshold, storage_config, administrative_divisions_patt, organization_patt,
#                  stop_words_machine):
#         self.num_perm = num_perm
#         self.threshold = threshold
#         self._storage = storage_config
#         self.data_ = []
#         self.lsh = AsyncMinHashLSH(storage_config=self._storage, threshold=self.threshold, num_perm=self.num_perm)
#         self.administrative_divisions_patt = administrative_divisions_patt
#         self.organization_patt = organization_patt
#         self.stop_words_machine = stop_words_machine

#     def batch_minhash(self, anchor_names):
#         self.data_ = [(anchor,
#                        min_hash_enterprise_name(anchor, self.administrative_divisions_patt, self.organization_patt,
#                                                 self.stop_words_machine, self.num_perm))
#                       for anchor in anchor_names]
#         return self

#     async def store_minhash(self):
#         async with self.lsh as lsh:
#             async with lsh.insertion_session(batch_size=10000) as session:
#                 fs = (session.insert(key, minhash, check_duplication=False) for key, minhash in self.data_)
#             await asyncio.gather(*fs)

#     async def query_one(self, name):
#         hash_value = min_hash_enterprise_name(name, self.administrative_divisions_patt, self.organization_patt,
#                                               self.stop_words_machine, self.num_perm)
#         return await self.lsh.query(hash_value)

#     async def query_hash(self, hash_value, lsh):
#         return await lsh.query(hash_value)

#     async def query_many(self, names):
#         tasks = []
#         hash_values = [min_hash_enterprise_name(name, self.administrative_divisions_patt, self.organization_patt,
#                                                 self.stop_words_machine, self.num_perm) for name in names]
#         async with self.lsh as lsh:
#             for hash_ in hash_values:
#                 task = asyncio.ensure_future(self.query_hash(hash_, lsh))
#                 tasks.append(task)
#         return await asyncio.gather(*tasks)

#     async def remove_names(self, names_to_remove):
#         async with self.lsh as lsh:
#             async with lsh.delete_session(batch_size=5) as session:
#                 fs = (session.remove(key) for key in names_to_remove)
#                 await asyncio.gather(*fs)

#     async def close(self):
#         await self.lsh.close()


if __name__ == '__main__':
    db = DBconnector(DBType='mysql', host="172.16.90.83", port=8066, db="csmsdb", user="bee", password="Llsbee!", url=None)
    data = db.query("""
                    SELECT
                    DISTINCT sellertaxno
                    FROM rz_multi_sellertaxno
                    WHERE newest_sellertaxno in
                                           (SELECT
                                            DISTINCT newest_sellertaxno
                                            FROM rz_multi_sellertaxno
                                            WHERE sellertaxno in
                                                               (
                                                               SELECT DISTINCT sellertaxno
                                                               from rz_invoice
                                                               WHERE
                                                               sellername = '{supplier_name}'
                                                               OR
                                                               sellertaxno = '{supplier_tax_number}'
                                                               )
                                           )""".format(supplier_name="汕头市亿立电讯有限公司", supplier_tax_number=None)).as_pandas().getQueryResult()
    print(data["sellertaxno"].unique())
