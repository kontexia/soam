#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Tuple, Set
import networkx as nx
import time
from copy import deepcopy
import plotly.graph_objects as go
from src.sdr import SDR


# network x edges returns a tuple in the following format
# (EDGE_SOURCE, EDGE_TARGET, EDGE_KEY, EDGE_PROP)
#
EDGE_SOURCE = 0
EDGE_TARGET = 1
EDGE_KEY = 2
EDGE_PROP = 3

# Edge Key Tuple constants
# edge keys are tuples with the following format:
# (EDGE_TYPE, EDGE_EXPIRY)
# these constants define the order in the tuple
#
EDGE_TYPE = 0
EDGE_EXPIRY = 1

Node_Type_Type = str
Node_Uid_Type = str
Edge_Type = str
Node_Type = Tuple[Node_Type_Type, Node_Uid_Type]


class AMFGraph(nx.MultiDiGraph):
    def __init__(self, incoming_graph_data: Optional[nx.MultiDiGraph] = None, **attr):
        """
        class to implement a multi directional network graph

        :param incoming_graph_data: another graph to initialise from
        :param attr:
        """
        super().__init__(incoming_graph_data, **attr)

    def set_edge(self, source: Node_Type, target: Node_Type, edge: Edge_Type,
                 prob: float = 1.0,
                 numeric: Optional[float] = None,
                 numeric_min: Optional[float] = None,
                 numeric_max: Optional[float] = None,
                 timestamp: Optional[float] = None,
                 **properties) -> None:
        """
        adds an edge to the graph. If source or target nodes do not exist then they will be created with appropriate properties

        :param source: tuple defining the source node type and uid i.e. (node_type, node_uid)
        :param target: tuple defining the target node type and uid i.e. (node_type, node_uid)
        :param edge: the edge type
        :param prob: the probability of this edge
        :param numeric: a real value associated with this edge
        :param numeric_min: the minimum that numeric can be
        :param numeric_max: the maximum that numeric can be
        :param timestamp: posix timestamp that defines when this edge is created - if None then current system time is used
        :param properties: provide named properties to add to the edge
        :return: None
        """

        # if timestamp not given then get current system timestamp
        #
        if timestamp is not None:
            ts = timestamp
        else:
            ts = time.time()

        # add source and target nodes if necessary
        #
        if source not in self:
            self.add_node(source, _created_ts=ts, _updated=True)

        if target not in self:
            self.add_node(target, _created_ts=ts, _updated=True)

        # the edge key used to identify this edge - the order is important and needs to be inline with edge key tuple constants defined above
        #
        edge_key = (edge, None)

        # remove exiting edge if it exists
        #
        if self.has_edge(source, target, key=edge_key):
            self.remove_edge(source, target, key=edge_key)

        # create a dictionary to hold other data associated with edge
        # note that '$' indicates special properties and distinguishes them from names of properties provided by library user
        #
        edge_properties = {'_created_ts': ts,
                           '_updated': True,             # flag indicating if it has been updated since last saved or restored
                           '_prob': prob,
                           '_numeric': numeric,
                           '_numeric_min': numeric_min,
                           '_numeric_max': numeric_max
                           }

        # allow properties to override settings
        #
        edge_properties.update(properties)
        self.add_edge(source, target, key=edge_key, **edge_properties)

    def update_edge(self, source: Node_Type, target: Node_Type, edge: Edge_Type,
                    prob: Optional[float] = None,
                    numeric: Optional[float] = None,
                    numeric_min: Optional[float] = None, numeric_max: Optional[float] = None,
                    timestamp: Optional[float] = None,
                    **properties) -> None:
        """
        performs an audited update of an edge - any existing edge is first expired by setting the expired timestamp and then a new edge version is created with the new data

        :param source: tuple defining the source node type and uid i.e. (<node_type>, <node_uid>)
        :param target: tuple defining the target node type and uid i.e. (<node_type>, <node_uid>)
        :param edge: the edge type
        :param prob: <float> the probability of this edge
        :param numeric: <float> a real value associated with this edge
        :param numeric_min: <float> the minimum that numeric can be
        :param numeric_max: <float> the maximum that numeric can be
        :param timestamp: posix timestamp that defines when this edge is updated - if None then current system time is used
        :param properties: provide named properties to add to the edge
        :return: None
        """

        # if timestamp not given then get current system timestamp
        #
        if timestamp is not None:
            ts = timestamp
        else:
            ts = time.time()

        # define the existing edge key - order is important and must be inline with edge key tuple constants defined above
        #
        exist_edge_key = (edge, None)

        # if edge already exists then create 'expired' version before adding 'new' version
        #
        if self.has_edge(source, target, key=exist_edge_key):

            prev_properties = deepcopy(self[source][target][exist_edge_key])
            prev_properties['_updated'] = True

            # define the expired edge key - order is important and must be inline with edge key tuple constants defined above
            #
            expired_edge_key = (edge, ts)
            self.add_edge(source, target, key=expired_edge_key, **prev_properties)

            # get the previous settings for key properties
            #
            if prob is None:
                prob = prev_properties['_prob']

            if numeric is None:
                numeric = prev_properties['_numeric']

            if numeric_min is None:
                numeric_min = prev_properties['_numeric_min']

            if numeric_max is None:
                numeric_max = prev_properties['_numeric_max']

        # overwrite existing edge
        #
        self.set_edge(source=source, target=target, edge=edge,
                      prob=prob, numeric=numeric, numeric_min=numeric_min, numeric_max=numeric_max,
                      timestamp=ts, **properties)

    def expire_edge(self, source: Node_Type, target: Node_Type, edge: Edge_Type,
                    timestamp: Optional[float] = None,
                    **properties) -> None:
        """
        'expire' an edge by performing a the equivalent of a soft delete

        :param source: tuple defining the source node type and uid i.e. (node_type, node_uid)
        :param target: tuple defining the target node type and uid i.e. (node_type, node_uid)
        :param edge: the edge type
        :param timestamp: posix timestamp that defines when this edge is expired - if None then current system time is used
        :param properties: provide named properties to add to the edge
        :return: None
        """

        # define the existing edge key - order is important and must be inline with edge key tuple constants defined above
        #
        exist_edge_key = (edge, None)

        # only do something if the edge exists
        #
        if self.has_edge(source, target, key=exist_edge_key):

            # if timestamp not given then get current system timestamp
            #
            if timestamp is not None:
                ts = timestamp
            else:
                ts = time.time()

            # get a copy of the existing edge attributes, set the expired_ts and add in any other attributes provided
            #
            prev_properties = deepcopy(self[source][target][exist_edge_key])

            # remove the existing unexpired edge
            #
            self.remove_edge(source, target, key=exist_edge_key)

            # add any properties provided
            #
            prev_properties.update(properties)
            prev_properties['_updated'] = True

            # define the expired edge key - order is important and must be inline with edge key tuple constants defined above
            #
            expired_edge_key = (edge, ts)

            # create the expired edge with previous properties
            #
            self.add_edge(source, target, key=expired_edge_key, **prev_properties)

    def set_node(self,
                 node: Node_Type,
                 node_attr: Optional[dict] = None,
                 node_prop: Optional[dict] = None,
                 edge_prop: Optional[dict] = None,
                 timestamp: Optional[float] = None,
                 stemmer_func=None) -> None:
        """

        adds a specified node and creates edges to any node attributes provided. Will also add edge to the stemmed version of the node if a stemmer function is provided

        :param node: tuple defining the node type and uid i.e. (node_type, node_uid)
        :param node_attr: a dictionary of attributes to create for this node.
                            key must have format: (edge uid,  (target_node_type, target_node_uid))
                            value is a dict {'prob': edge probability,
                                            'numeric': a numeric value for edge,
                                            'numeric_min': the minimum value for numeric,
                                            'numeric_max': the maximum value for numeric}
        :param node_prop: a dictionary of immutable properties to add to the node
        :param edge_prop: a dictionary of immutable properties to add to every edge created
        :param timestamp: posix timestamp that defines when this node is created - if None then current system time is used
        :param stemmer_func: a stemmer function that takes a string and returns a stemmed version of string
        :return: None
        """

        # if timestamp not given then get current system timestamp
        #
        if timestamp is not None:
            ts = timestamp
        else:
            ts = time.time()

        # create the node if it doesnt already exist
        #
        if node_prop is not None:
            self.add_node(node, _created_ts=ts, _updated=True, **node_prop)

        # create the stemmed version of the node if required
        #
        if stemmer_func is not None:

            # the target node is the stemmed version
            #
            target = ('expression', stemmer_func(node[1]))
            if edge_prop is not None:
                self.set_edge(source=node, target=target, edge='has_stem', timestamp=timestamp, **edge_prop)
            else:
                self.set_edge(source=node, target=target, edge='has_stem', timestamp=timestamp)

        # create edges to node attributes if required
        #
        if node_attr is not None:

            for key in node_attr:
                # unpack the sdr_key
                #
                edge = key[0]
                target = key[1]

                if 'prob' in node_attr[key]:
                    prob = node_attr[key]['prob']
                else:
                    prob = 1.0

                if 'numeric' in node_attr[key]:
                    numeric = node_attr[key]['numeric']
                else:
                    numeric = None

                if 'numeric_min' in node_attr[key]:
                    numeric_min = node_attr[key]['numeric_min']
                else:
                    numeric_min = None

                if 'numeric_max' in node_attr[key]:
                    numeric_max = node_attr[key]['numeric_max']
                else:
                    numeric_max = None

                if edge_prop is not None:
                    self.set_edge(source=node, target=target, edge=edge,
                                  prob=prob, numeric=numeric, numeric_min=numeric_min, numeric_max=numeric_max,
                                  timestamp=ts, **edge_prop)
                else:
                    self.set_edge(source=node, target=target, edge=edge,
                                  prob=prob, numeric=numeric, numeric_min=numeric_min, numeric_max=numeric_max,
                                  timestamp=ts)

                if stemmer_func is not None:
                    source = target
                    target = ('expression', stemmer_func(target[1]))

                    # add an edge to the stem node
                    #
                    if edge_prop is not None:
                        self.set_edge(source=source, target=target, edge='has_stem', prob=1.0, timestamp=ts, **edge_prop)
                    else:
                        self.set_edge(source=source, target=target, edge='has_stem', prob=1.0, timestamp=ts)

    def update_node(self, node: Node_Type,
                    upsert_attr: Optional[dict] = None,
                    expire_attr: Optional[set] = None,
                    edge_prop: Optional[dict] = None,
                    timestamp: Optional[float] = None):
        """
        updates / expires the attributes of a node

        :param node: tuple defining the node type and uid i.e. (node_type>, <node_uid)
        :param upsert_attr: a dictionary of attributes to update for this node.
                            key must have format: (edge uid,  (target_node_type, target_node_uid))
                            value is a dict {'prob': edge probability,
                                            'numeric': a numeric value for edge,
                                            'numeric_min': the minimum value for numeric,
                                            'numeric_max': the maximum value for numeric}
        :param expire_attr: a set of the attributes to expire
        :param edge_prop: a dictionary of properties to add to every edge created
        :param timestamp: posix timestamp that defines when this node is created - if None then current system time is used
        :return:
        """
        if upsert_attr is not None:

            for key in upsert_attr:
                edge = key[0]
                target = key[1]

                if 'prob' in upsert_attr[key]:
                    prob = upsert_attr[key]['prob']
                else:
                    prob = None

                if 'numeric' in upsert_attr[key]:
                    numeric = upsert_attr[key]['numeric']
                else:
                    numeric = None

                if 'numeric_min' in upsert_attr[key]:
                    numeric_min = upsert_attr[key]['numeric_min']
                else:
                    numeric_min = None

                if 'numeric_max' in upsert_attr[key]:
                    numeric_max = upsert_attr[key]['numeric_max']
                else:
                    numeric_max = None

                if edge_prop is not None:
                    self.update_edge(source=node, target=target, edge=edge,
                                     prob=prob, numeric=numeric, numeric_min=numeric_min, numeric_max=numeric_max,
                                     timestamp=timestamp, **edge_prop)
                else:
                    self.update_edge(source=node, target=target, edge=edge,
                                     prob=prob, numeric=numeric, numeric_min=numeric_min, numeric_max=numeric_max,
                                     timestamp=timestamp)

        if expire_attr is not None:

            for key in expire_attr:
                edge = key[0]
                target = key[1]

                if edge_prop is not None:
                    self.expire_edge(source=node, target=target, edge=edge, timestamp=timestamp, **edge_prop)
                else:
                    self.expire_edge(source=node, target=target, edge=edge, timestamp=timestamp)

    def get_data_to_persist(self, persist_all_override: bool = False):
        """
        returns dictionaries of nodes and edges to persist and edges to expire
        :param persist_all_override: if false only nodes and edges that require persisting will be added - else all are included
        :return: nodes_to_persist dict with following format: {<node collection>: [{_key: uid of node, property: property_value]}
                 edges_to_persist dict with following format: {<edge type>: [{_key: uid of node, property: property_value]}
                 expired_edges dict with following format:
        """
        nodes_to_persist = {}

        for node in self.nodes(data=True):
            if persist_all_override or node[1]['_updated']:
                if node[0][0] not in nodes_to_persist:
                    nodes_to_persist[node[0][0]] = []
                nodes_to_persist[node[0][0]].append({'_key': str(node[0][1]).replace('\'', '').replace('(', '').replace(')', '').replace(' ', '_'),
                                                     '_node_type': node[0][0],      # will need this field to reconstruct node on restore
                                                     '_node_uid': node[0][1],       # will need this field to reconstruct node on restore
                                                     **{prop: node[1][prop] for prop in node[1] if prop != '_updated'}})
                node[1]['_updated'] = False

        edges_to_persist = {}
        for edge in self.edges(keys=True, data=True):
            source = edge[EDGE_SOURCE]
            target = edge[EDGE_TARGET]
            edge_key = edge[EDGE_KEY]
            edge_prop = edge[EDGE_PROP]

            if persist_all_override or edge_prop['_updated']:

                # assume edge_key is tuple (EDGE_TYPE, EDGE_EXPIRY)
                #
                if edge_key[EDGE_TYPE] not in edges_to_persist:
                    edges_to_persist[edge_key[EDGE_TYPE]] = []

                edges_to_persist[edge_key[EDGE_TYPE]].append({'_key': '{}:{}:{}:{}:{}:{}'.format(source[0], str(source[1]).replace('\'', '').replace('(', '').replace(')', '').replace(' ', '_'),
                                                                                                 edge_key[EDGE_TYPE], edge_key[EDGE_EXPIRY],
                                                                                                 target[0], str(target[1]).replace('\'', '').replace('(', '').replace(')', '').replace(' ', '_')),
                                                              '_from': '{}/{}'.format(source[0], str(source[1]).replace('\'', '').replace('(', '').replace(')', '').replace(' ', '_')),
                                                              '_to': '{}/{}'.format(target[0], str(target[1]).replace('\'', '').replace('(', '').replace(')', '').replace(' ', '_')),
                                                              '_source_type': source[0],
                                                              '_source_uid': source[1],
                                                              '_target_type': target[0],
                                                              '_target_uid': target[1],
                                                              '_edge': edge_key[EDGE_TYPE],
                                                              '_expired_ts': edge_key[EDGE_EXPIRY],
                                                              **{prop: edge_prop[prop] for prop in edge_prop if prop != '_updated'}})
                edge_prop['_updated'] = False

        return nodes_to_persist, edges_to_persist

    def restore_edges(self, edges):
        for edge in edges:
            edge_prop = {prop: edge[prop]
                         for prop in edge
                         if prop not in ['_id', '_key', '_rev', '_from', '_to', '_source_uid', '_target_uid', '_source_type', '_target_type', '_edge', '_expired_ts']}
            self.add_edge((edge['_source_type'], edge['_source_uid']), (edge['_target_type'], edge['_target_uid']),
                          key=(edge['_edge'], edge['_expired_ts']),
                          _updated=False, **edge_prop)

    def restore_nodes(self, nodes):
        for node in nodes:
            node_prop = {prop: node[prop] for prop in node if prop not in ['_id', '_key', '_rev', '_node_uid', '_node_type']}
            self.add_node((node['_node_type'], node['_node_uid']), _updated=False, **node_prop)

    def _check_filter(self, item_to_check: tuple, filter_to_apply: dict):
        """
        recursive function that applies the filter_to_apply expression and returns either True or False
        :param item_to_check:   on first entry expect tuple of data with following formats:
                                    (((<source type>, <source uid>), <source property dict>), ((<target type>, <target uid>), <target property dict>), (<edge uid>, <edge property dict>))
                                on subsequent entries expect format:
                                    ((<source type>, <source uid>), <source property dict>)
                                    ((<target type>, <target uid>), <target property dict>)
                                    (<edge uid>, <edge property dict>))

        :param filter_to_apply: a (potentially nested) dictionary that defines the filter expression using key words:

                                {'$and': [<query expressions>]}     logical AND of list of expressions - which can be nested
                                {'$or': [<query expressions>]}      logical OR of list of expressions - which can be nested

                                {'$not': {<expression>}}            returns true if <expression> false

                                {'$source': {<expression>}}         expression to be applied to the source nodes
                                {'$target': {<expression>}}         expression to be applied to the target nodes
                                {'$edge': {<expression>}}           expression to be applied to the edges

                                {'$uid' : {<expression>}}           expression to be applied to the uid of either the source, target or edge
                                {'$type' : {<expression>}}          expression to be applied to the type of either the source or target
                                {'$probability' : {<expression>}}   expression to be applied to the probability property of an edge
                                {'$numeric' : {<expression>}}       expression to be applied to the numeric value property of an edge
                                {'$numeric_min' : {<expression>}}   expression to be applied to the min numeric property of an edge
                                {'$numeric_max' : {<expression>}}   expression to be applied to the max numeric property of an edge
                                {'$created_ts' : {<expression>}}    expression to be applied to the created timestamp property of an edge or node
                                {'$expired_ts' : {<expression>}}    expression to be applied to the expired timestamp property of an edge
                                {'$eq': value}                      property equal to <value>
                                {'$ne': value}                      property not equal to <value>
                                {'$lt': value}                      property less than <value>
                                {'$lte': value}                     property less than or equal to <value>
                                {'$gt': value}                      property greater than <value>
                                {'$gte': value}                     property greater than or equal to <value>
                                {'$in_value': value}                <value> in property
                                {'$value_in': [value_1,...]}        property in list of <value>

        :return: <bool> True if filter_to_apply expression evaluates to True else False
        """

        result = None

        for key in filter_to_apply.keys():
            if key == '$and':
                q_result = True
                for and_query in filter_to_apply['$and']:
                    q_result = self._check_filter(item_to_check, and_query)
                    if not q_result:
                        break

            elif key == '$or':
                q_result = False
                for or_query in filter_to_apply['$or']:
                    q_result = self._check_filter(item_to_check, or_query)
                    if q_result:
                        break

            elif key == '$not':
                q_result = not self._check_filter(item_to_check, filter_to_apply['$not'])

            elif key == '$source':
                q_result = self._check_filter(item_to_check[0], filter_to_apply['$source'])
            elif key == '$target':
                q_result = self._check_filter(item_to_check[1], filter_to_apply['$target'])
            elif key == '$edge':
                q_result = self._check_filter(item_to_check[2], filter_to_apply['$edge'])
            else:

                if key == '$uid':
                    # if item_to_check[0] is a tuple then return second entry
                    #
                    if isinstance(item_to_check[0], tuple):
                        # if len is 2 then must be a node and uid is in 2nd slot
                        #
                        if len(item_to_check[0]) == 2:
                            value = item_to_check[0][1]
                        else:
                            # else must be an edge and uid id in 1st slot
                            #
                            value = item_to_check[0][0]
                    else:
                        value = item_to_check[0]
                elif key == '$type':
                    # if item_to_check[0] is a tuple then return first entry
                    #
                    if isinstance(item_to_check[0], tuple):
                        value = item_to_check[0][0]
                    else:
                        value = item_to_check[0]
                elif key == '$probability':
                    value = item_to_check[1]['_prob']
                elif key == '$numeric':
                    if '_numeric' in item_to_check[1]:
                        value = item_to_check[1]['_numeric']
                    else:
                        break
                elif key == '$numeric_min':
                    if '_numeric_min' in item_to_check[1]:
                        value = item_to_check[1]['_numeric_min']
                    else:
                        break
                elif key == '$numeric_max':
                    if '_numeric_max' in item_to_check[1]:
                        value = item_to_check[1]['_numeric_max']
                    else:
                        break
                elif key == '$created_ts':
                    value = item_to_check[1]['_created_ts']
                elif key == '$expired_ts' and isinstance(item_to_check[0], tuple):
                    # expired_ts held in the edge_key tuple
                    #
                    value = item_to_check[0][2]
                elif key in item_to_check[1]:
                    value = item_to_check[1][key]
                else:
                    # if we don't recognise the key then error
                    #
                    break

                if '$eq' in filter_to_apply[key]:
                    q_result = (value == filter_to_apply[key]['$eq'])

                elif '$ne' in filter_to_apply[key]:
                    q_result = (value != filter_to_apply[key]['$ne'])

                elif '$lt' in filter_to_apply[key]:
                    q_result = (value < filter_to_apply[key]['$lt'])

                elif '$lte' in filter_to_apply[key]:
                    q_result = (value <= filter_to_apply[key]['$lte'])

                elif '$gt' in filter_to_apply[key]:
                    q_result = (value > filter_to_apply[key]['$gt'])

                elif '$gte' in filter_to_apply[key]:
                    q_result = (value >= filter_to_apply[key]['$gte'])

                elif '$in_value' in filter_to_apply[key]:
                    q_result = (filter_to_apply[key]['$in_value'] in value)

                elif '$value_in' in filter_to_apply[key]:
                    q_result = (value in filter_to_apply[key]['$value_in'])
                else:
                    q_result = False

            if result is None:
                result = q_result
            else:
                result = result and q_result

            if not result:
                break

        return result

    def filter_sub_graph(self, query: dict):
        """
        given a query dictionary return a filtered sub graph

        :param query: a (potentially nested) dictionary that defines the filter expression using key words:

                {'$and': [<query expressions>]}     logical AND of list of expressions - which can be nested
                {'$or': [<query expressions>]}      logical OR of list of expressions - which can be nested

                {'$not': {<expression>}}            returns true if <expression> false

                {'$source': {<expression>}}         expression to be applied to the source nodes
                {'$target': {<expression>}}         expression to be applied to the target nodes
                {'$edge': {<expression>}}           expression to be applied to the edges

                {'$uid' : {<expression>}}           expression to be applied to the uid of either the source, target or edge
                {'$type' : {<expression>}}          expression to be applied to the type of either the source or target
                {'$probability' : {<expression>}}   expression to be applied to the probability property of an edge
                {'$numeric' : {<expression>}}       expression to be applied to the numeric value property of an edge
                {'$numeric_min' : {<expression>}}      expression to be applied to the min numeric property of an edge
                {'$numeric_max' : {<expression>}}      expression to be applied to the max numeric property of an edge
                {'$created_ts' : {<expression>}}    expression to be applied to the created timestamp property of an edge or node
                {'$expired_ts' : {<expression>}}    expression to be applied to the expired timestamp property of an edge
                {'$eq': value}                      property equal to <value>
                {'$ne': value}                      property not equal to <value>
                {'$lt': value}                      property less than <value>
                {'$lte': value}                     property less than or equal to <value>
                {'$gt': value}                      property greater than <value>
                {'$gte': value}                     property greater than or equal to <value>
                {'$in_value': value}                <value> in property
                {'$value_in': [value_1,...]}        property in list of <value>

        :return: the filtered sub graph
        """
        edges = {(source, target, edge)
                 for source in self
                 for target in self[source]
                 for edge in self[source][target]
                 if self._check_filter(item_to_check=((source, self.nodes[source]), (target, self.nodes[target]), (edge, self[source][target][edge])), filter_to_apply=query)}

        result_graph = self.edge_subgraph(edges)
        return result_graph

    def _get_attr_sdr(self, node, nos_hops, exclude_nodes, exclude_edges=None, generalised_node_name=None) -> SDR:
        """
        recursive method to extract all nodes connected to a node
        :param node: the node to find connections from
        :param nos_hops: the number of hops from node to find connections
        :param exclude_edges: edges to ignore
        :param exclude_nodes: nodes not to revisit
        :param generalised_node_name: a substitute for node name
        :return: SDR
        """

        sdr = SDR()
        for target in self[node]:
            for edge in self[node][target]:
                if exclude_edges is None or edge not in exclude_edges:
                    edge_prob = self[node][target][edge]['_prob']
                    edge_numeric = None
                    edge_min = None
                    edge_max = None
                    if '_numeric' in self[node][target][edge]:
                        edge_numeric = self[node][target][edge]['_numeric']
                    if '_numeric_min' in self[node][target][edge]:
                        edge_min = self[node][target][edge]['_numeric_min']
                    if '_numeric_max' in self[node][target][edge]:
                        edge_max = self[node][target][edge]['_numeric_max']

                    if generalised_node_name is not None:
                        src = (node[0], generalised_node_name)
                    else:
                        src = node
                    sdr.set_item(source_node=src, edge=edge[EDGE_TYPE], target_node=target, probability=edge_prob, numeric=edge_numeric, numeric_min=edge_min, numeric_max=edge_max)

                    if nos_hops - 1 > 0 and target not in exclude_nodes:
                        sdr.update(self._get_attr_sdr(node=target, nos_hops=nos_hops - 1, exclude_nodes={target, *exclude_nodes}, exclude_edges=exclude_edges))
        return sdr

    def get_node_sdr(self, node: Node_Type,
                     nos_hops: int = 1,
                     exclude_edges: Set[Edge_Type] = None,
                     target_node_edge: str = None,
                     generalised_node_name: str = '*') -> SDR:
        """
        method that returns an SDR of a sub graph connected to node

        :param node: the node to extract the sub graph for
        :param nos_hops: the number of hops from node to search
        :param exclude_edges: edge types to exclude from the extract
        :param target_node_edge: If not None then node will be included in the SDR, connected to the generalised node via an edge with name 'target_node_name'
        :param generalised_node_name: a string to represent the generalised name of parameter node
        :return: SDR
        """

        sdr = SDR()
        if node in self:

            # add the generalised edge to the target node if required
            #
            if target_node_edge is not None:
                sdr.set_item(source_node=(node[0], generalised_node_name), edge=target_node_edge, target_node=node, probability=1.0, numeric=None, numeric_min=None, numeric_max=None)

            # get any attributes
            #
            sdr.update(self._get_attr_sdr(node=node, nos_hops=nos_hops, generalised_node_name=generalised_node_name, exclude_nodes={node}, exclude_edges=exclude_edges))

        return sdr

    def get_target_node(self, node: Node_Type, edge_filter_func=None):
        result = None
        if node in self:
            result = []
            for target in self[node]:
                if edge_filter_func is not None:
                    for edge in self[node][target]:
                        if edge_filter_func(edge):
                            result.append(target)
                else:
                    result.append(target)
        return result

    def __str__(self):
        txt = super().__str__()
        txt = txt + '\nNodes:\n'
        for node in self.nodes(data=True):
            txt = txt + str(node) + '\n'
        txt = txt + '\nEdges:\n'
        for edge in self.edges(keys=True, data=True):
            txt = txt + str(edge) + '\n'
        return txt

    def get_cores(self):
        dg = nx.DiGraph(self)
        dg.remove_edges_from(nx.selfloop_edges(dg))
        decomposition = nx.core_number(dg)
        return decomposition

    def plot(self, dimension=2, weight_field='_prob', node_filter_func=None, edge_filter_func=None):

        # get the xyz coordinates using the spring force algorthm
        #
        coords = nx.spring_layout(self, dim=dimension, weight=weight_field)

        x = []
        y = []
        z = []
        edge_x = []
        edge_y = []
        edge_z = []

        node_labels = []
        edge_labels = []
        colors = []
        types = {}
        color_id = 0

        nodes_to_display = []

        # get nodes to display given the various filters specified
        #
        for node in coords:
            if node_filter_func is not None:
                if node_filter_func(node):
                    nodes_to_display.append(node)
                    for target in self[node]:
                        if edge_filter_func is not None:
                            include = True
                            for edge in self[node][target]:
                                if not edge_filter_func(edge):
                                    include = False
                                    break
                            if include:
                                nodes_to_display.append(target)
                        else:
                            nodes_to_display.append(target)
            else:
                nodes_to_display.append(node)

        for node in nodes_to_display:
            node_labels.append('{} : {}'.format(node[0], node[1]))

            if node[0] not in types:
                types[node[0]] = color_id
                color_id += 1
            colors.append(types[node[0]])
            x.append(coords[node][0])
            y.append(coords[node][1])
            if dimension == 3:
                z.append(coords[node][2])

            for target in self[node]:
                h_txt = None
                if edge_filter_func is not None:
                    for edge in self[node][target]:
                        if edge_filter_func(edge):
                            edge_x.append(coords[node][0])
                            edge_x.append(coords[target][0])
                            edge_x.append(None)
                            edge_y.append(coords[node][1])
                            edge_y.append(coords[target][1])
                            edge_y.append(None)
                            if dimension == 3:
                                edge_z.append(coords[node][2])
                                edge_z.append(coords[target][2])
                                edge_z.append(None)
                            if '_numeric' in self[node][target][edge]:
                                numeric = self[node][target][edge]['_numeric']
                            else:
                                numeric = None
                            h_txt = 'Src: {} : {} Trg: {} : {}<br>Prob: {}<br>Numeric: {}'.format(node[0], node[1],
                                                                                                  target[0], target[1],
                                                                                                  self[node][target][edge]['_prob'], numeric)
                else:
                    edge_x.append(coords[node][0])
                    edge_x.append(coords[target][0])
                    edge_x.append(None)
                    edge_y.append(coords[node][1])
                    edge_y.append(coords[target][1])
                    edge_y.append(None)
                    if dimension == 3:
                        edge_z.append(coords[node][2])
                        edge_z.append(coords[target][2])
                        edge_z.append(None)
                    for edge in self[node][target]:
                        if '_numeric' in self[node][target][edge]:
                            numeric = self[node][target][edge]['_numeric']
                        else:
                            numeric = None
                        h_txt = 'Src: {} : {} Trg: {} : {}<br>Prob: {}<br>Numeric: {}'.format(node[0], node[1],
                                                                                              target[0], target[1],
                                                                                              self[node][target][edge]['_prob'], numeric)
                edge_labels.extend(['', h_txt, None])

        if dimension == 2:
            edge_scatter = go.Scatter(x=edge_x, y=edge_y, text=edge_labels, mode='lines', line=dict(width=1, color='grey'))

            node_scatter = go.Scatter(x=x, y=y, text=node_labels, mode='markers+text', marker=dict(size=20, color=colors, opacity=0.7))
            fig = go.Figure(data=[edge_scatter, node_scatter])
            fig.update_layout(width=900, height=900, title=dict(text='AMFGraph'),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

        else:
            edge_scatter = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, text=edge_labels, mode='lines', line=dict(width=1, color='grey'))

            node_scatter = go.Scatter3d(x=x, y=y, z=z, text=node_labels, mode='markers+text', marker=dict(size=20, color=colors, opacity=0.7))

            fig = go.Figure(data=[node_scatter, edge_scatter])
            fig.update_layout(width=900, height=900, title=dict(text='AMFGraph'),
                              scene=dict(
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        fig.show()



if __name__ == '__main__':

    g = AMFGraph()

    #  identify the node with a tuple with format (<NODE_TYPE>, <NODE_UID>)
    #
    node_id = ('Trade', 'XYZ_123')

    # define static properties of the node as a simple dictionary
    #
    node_properties = {'other_id': 123}

    # define any 'attributes' of Trade with a dictionary with the following format:
    #
    # edge_to attribute_key = (<RELATIONSHIP>, <ATTRIBUTE_NODE_ID>)
    # edge properties = either <WEIGHT> or (<WEIGHT>, <VALUE>, <MIN_VALUE>, <MAX_VALUE>)
    #
    node_attr = {('has_client', ('client', 'abc ltd')): {'prob': 1.0},  # probability of edge is 1.0
                 ('has_platform', ('platform', 'electronic')): {'prob': 1.0},  # probability of edge is 1.0
                 ('has_product', ('product', 'swap')): {'prob': 1.0},  # probability of edge is 1.0

                 # here we set a probability of edge to 1.0, we also associate numeric value 800mio, minimum value of 0, max value of 1 bio to the edge
                 #
                 ('has_nominal', ('nominal', 'swap_nominal')): {'prob': 1.0, 'numeric': 800000000, 'numeric_min': 0, 'numeric_max': 1000000000}
                 }


    # one can specify extra properties for every edge created
    #
    edge_properties = {'other_id': 123}

    # add the node
    #
    g.set_node(node=node_id, node_attr=node_attr, node_prop=node_properties, edge_prop=edge_properties, timestamp=None)

    #  you can print the graph
    #
    print(g)

    # and you can plot the resulting graph
    #
    g.plot(dimension=3)

    # define a dictionary of the attributes to be inserted/ updated
    #
    upsert_attr = {('has_nominal', ('nominal', 'swap_nominal')): {'numeric': 5000000}}

    # define a set of 'attribute relationships" to be expired
    #
    expire_attr = {('has_interest', ('interest', 'banking'), 0)}

    edge_properties = {'update_id': 321}

    g.update_node(node=node_id, upsert_attr=upsert_attr, expire_attr=expire_attr, edge_prop=edge_properties, timestamp=None)

    print(g)

    #  plot the graph filtering on expired_timestamp is None
    #
    g.plot(dimension=3, edge_filter_func=lambda x: x[1] is None)



