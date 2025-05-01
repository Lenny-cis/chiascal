# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:30:29 2022

@author: Lenny
"""

import pandas as pd
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


class CorrGraphSelector(TransformerMixin, BaseEstimator):
    """连通图筛选相关变量."""

    def __init__(self, corr=0.8, method='spearman):
        self.corr = corr
    	self.method = method

	@FuncRunInfo(logger)
    def fit(self, X, y=None, iv_func=None):
        """训练."""
        logger.info('Start {} fit'.format(self.__class__.__name__))
        num_vars = [name for name, dtp in X.dtypes.items()
                    if pd.api.types.is_numeric_dtype(dtp)]
        if self.method == 'spearman':
            corr = X.loc[:, num_vars].rank().corr()
        else:
	        corr = X.loc[:, num_vars].corr(method=self.method)
        corr_g = nx.from_pandas_adjacency(corr.fillna(0))
        corr_g.remove_edges_from(nx.selfloop_edges(corr_g))
        _ = [corr_g.remove_edge(*e) for e, w in corr_g.edges().items()
             if w['weight'] < self.corr]
        if iv_func is None:
            iv_func = X.std
        node_iv = {key: {'ivval': val} for key, val in iv_func().items()}
        nx.set_node_attributes(corr_g, node_iv)
        conn_comps = [c for c in nx.connected_components(corr_g)]
        res_var = [sorted(nx.get_node_attributes(
            corr_g.subgraph(comp), 'ivval').items(), key=lambda x: x[1],
            reverse=True)[0][0] for comp in conn_comps]
        res_var.extend(X.columns.difference(num_vars))
        self.input_vars = X.columns.tolist()
        self.output_vars = res_var
        self.corr_mat = corr

    def transform(self, X):
        """应用."""
        trn_x_name = [x for x in X.columns if x in self.output_vars]
        return X.loc[:, trn_x_name]

	@property
    def rept(self):
        """Report DF."""
        in_num = len(self.input_vars)
        ou_num = len(self.output_vars)
        summ = pd.DataFrame.from_dict({
            'CORR': {'input': in_num, 'filter': in_num-ou_num,
                     'output': ou_num}},
            orient='index')
        return Report_tuple(summ, self.corr_mat)
