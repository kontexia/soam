#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Dict, Tuple, Union, Set
from copy import deepcopy
from math import pow, sqrt

# type Aliases
#

SGNodeType_Alias = str
""" type for node_type """

SGNodeUID_Alias = str
""" type for node_uid """

SGEdgeType_Alias = str
""" type for edge_type """

SGEdgeUID_Alias = Optional[str]
""" type for edge_uid"""

SGNode_Alias = Tuple[SGNodeType_Alias, SGNodeUID_Alias]
""" SGNode_Alias is a tuple of strings (node_type and node_uid) """

SGEdge_Alias = Tuple[SGEdgeType_Alias, SGEdgeUID_Alias]
""" SGEdge_Alias is a tuple (edge_type edge_uid) """

SGKey_Alias = str
""" SGKey_Alias is a string representation of (source_node, edge_type, target_node) """

SGValue_Alias = Dict[str, Union[str, float, None]]
""" SDRValue_Alias is a dictionary containing source node target node, edge, probability and numeric data """

SGSearchPor_Alias = Dict[SGKey_Alias, Dict[str, float]]


class SubGraph:

    def __init__(self, sub_graph=None):
        self.sdr: dict = {}

        if sub_graph is not None:
            if isinstance(sub_graph, SubGraph):
                self.sdr = deepcopy(sub_graph.sdr)
            elif isinstance(sub_graph, dict):
                self.sdr = deepcopy(sub_graph)

    def __contains__(self, sg_key: SGKey_Alias) -> bool:
        """
        method to check if an sdr_key exists in the sdr
        :param sg_key: edge key to check
        :return: True if it exists else False
        """

        return sg_key in self.sdr

    def __iter__(self) -> iter:
        """
        method to return an iterable of the sdr keys
        :return: iterable of self keys
        """
        return iter(self.sdr)

    def __getitem__(self, sg_key: SGKey_Alias) -> SGValue_Alias:
        """
        method to access the sdr edge attributes
        :param sg_key: the edge to return
        :return: the edge attributes {source_type:, source_uid:, target_type: , target_uid:, edge_type:, edge_uid:, prob:, numeric:}
        """
        return self.sdr[sg_key]

    def __setitem__(self, key: SGKey_Alias, value):
        self.sdr[key] = value

    def __delitem__(self, sg_key: SGKey_Alias) -> bool:
        """
        method to delete an edge from the sdr
        :param sdr_key: the edge to delete
        :return: True of deleted else false
        """
        result = False
        if sg_key in self.sdr:
            del self.sdr[sg_key]
            result = True
        return result

    def __len__(self):
        """
        method returns the number of edges in sdr
        :return: the length of the sdr
        """
        return len(self.sdr)

    def set_item(self,
                 source_node: SGNode_Alias,
                 edge: SGEdge_Alias,
                 target_node: SGNode_Alias,
                 probability: float = 1.0,
                 numeric: Optional[float] = None) -> SGKey_Alias:
        """
        method to set the sdr attributes

        :param source_node: tuple of (source_type, source uid)
        :param edge: edge_type
        :param target_node: tuple of (target_type, target uid)
        :param probability: probability of the edge
        :param numeric: numeric value associated with edge
        :return: None
        """
        sg_key = '{}:{}:{}:{}:{}:{}'.format(source_node[0], source_node[1], edge[0], edge[1], target_node[0], target_node[1])
        self.sdr[sg_key] = {'source_type': source_node[0], 'source_uid': source_node[1],
                            'target_type': target_node[0], 'target_uid': target_node[1],
                            'edge_type': edge[0],
                            'edge_uid': edge[1],
                            'prob': probability,
                            'numeric': numeric
                            }
        return sg_key

    def get_edge_types(self) -> Set[SGEdgeType_Alias]:
        """
        method returns the edge types in sdr
        :return: set of edge types
        """
        edge_types = {self.sdr[sg_key]['edge_type'] for sg_key in self.sdr}
        return edge_types

    def update(self, sub_graph) -> None:
        """
        method to update and sdr
        :param sub_graph: the sub graph to update with
        :return: None
        """
        self.sdr.update(sub_graph.sdr)

    def get_dict(self) -> Dict[SGKey_Alias, SGValue_Alias]:
        """
        method to return a dict representation

        :return: dict of dicts: {edge_key: {'source_type' , 'source_uid' , 'edge_type' , 'edge_uid' , 'target_type' , 'target_uid', 'prob', 'numeric'}
        """
        return deepcopy(self.sdr)

    def calc_euclidean_distance(self,
                                sub_graph,
                                compare_edge_types: Set[SGEdgeType_Alias],
                                edge_type_weights: Optional[Dict] = None) -> Tuple[float, SGSearchPor_Alias]:
        """
        method to calculate the euclidean distance of two sdrs

        :param sub_graph: sub graph to compare to
        :param compare_edge_types: set of SDR edge types that will be compared in the algorithm - all others are ignored
        :param edge_type_weights: an optional dictionary mapping edge types to weights - if None then assumes weights are equal
        :return: typle containing distance and a por dictionary with the following structure:
                {<SDR edge key>: {'prob': < probability distance >, 'numeric': < numeric distance > }
        """

        por: SGSearchPor_Alias
        distance: float = 0.0
        neuron_id: int
        edges_to_compare: Set[SGEdgeType_Alias]
        edge: SGKey_Alias
        numeric_dist: float
        prob_dist: float

        por = {}

        # default edge type weights at equal
        #
        if edge_type_weights is None:
            edge_type_weights = {edge_type: 1.0 for edge_type in compare_edge_types}

        # get edges to compare as a union of sdr_a and sdr_b
        #
        edges_to_compare = ({edge for edge in sub_graph
                            if sub_graph[edge]['edge_type'] in compare_edge_types} |

                            {edge for edge in self
                             if self[edge]['edge_type'] in compare_edge_types})

        for edge in edges_to_compare:
            numeric_dist = 0.0

            if edge in self and edge in sub_graph:
                edge_type = self[edge]['edge_type']
                prob_dist = abs(self[edge]['prob'] - sub_graph[edge]['prob'])

                if self[edge]['numeric'] is not None and sub_graph[edge]['numeric'] is not None:
                    numeric_dist = abs(self[edge]['numeric'] - sub_graph[edge]['numeric'])

            elif edge in self:
                edge_type = self[edge]['edge_type']

                prob_dist = self[edge]['prob']
                if self[edge]['numeric'] is not None:
                    numeric_dist = self[edge]['numeric']

            else:
                edge_type = sub_graph[edge]['edge_type']
                prob_dist = sub_graph[edge]['prob']
                if sub_graph[edge]['numeric'] is not None:
                    numeric_dist = sub_graph[edge]['numeric']

            por[edge] = {'prob': prob_dist, 'numeric': numeric_dist, 'weight': edge_type_weights[edge_type]}

            # add the probability distance
            #
            distance += pow(prob_dist * edge_type_weights[edge_type], 2)

            # add the numeric distance
            #
            distance += pow(numeric_dist * edge_type_weights[edge_type], 2)

        distance = sqrt(distance)
        return distance, por

    def learn(self,
              sub_graph,
              learn_rate: float,
              learn_edge_types: Set[SGEdgeType_Alias],
              prune_threshold: float = 0.01) -> None:
        """
        method to learn from the incoming sub graph
        :param sub_graph: the sub graph to learn
        :param learn_rate: the rate of learning
        :param learn_edge_types: the edge types to be learnt
        :param prune_threshold: the probability threshold below which an edge is assumed to be zero and thus deleted
        :return: None
        """

        # if we haven't already learnt some edges learn_rate defaulted to max 1.0
        #
        exist_edges = {edge for edge in self if self[edge]['edge_type'] in learn_edge_types}
        if len(exist_edges) == 0:
            learn_rate = 1.0

        # get union of edges to update
        #
        edges_to_update = ({edge for edge in sub_graph
                            if sub_graph[edge]['edge_type'] in learn_edge_types} | exist_edges)

        edges_to_delete = set()
        for edge in edges_to_update:

            # if edge is in both sdrs
            #
            if edge in self and edge in sub_graph:
                self[edge]['prob'] += ((sub_graph[edge]['prob'] - self[edge]['prob']) * learn_rate)

                if self[edge]['prob'] > prune_threshold:
                    if self[edge]['numeric'] is not None:
                        self[edge]['numeric'] += ((sub_graph[edge]['numeric'] - self[edge]['numeric']) * learn_rate)
                else:
                    edges_to_delete.add(edge)

            # edge only in this sdr so weaken
            #
            elif edge in self:
                self[edge]['prob'] += (0.0 - self[edge]['prob']) * learn_rate

                if self[edge]['prob'] > prune_threshold:
                    if self[edge]['numeric'] is not None:
                        self[edge]['numeric'] += (0.0 - self[edge]['numeric']) * learn_rate
                else:
                    edges_to_delete.add(edge)

            # edge only in new sdr so add
            #
            else:
                edge_item = deepcopy(sub_graph[edge])
                edge_item['prob'] = edge_item['prob'] * learn_rate

                if edge_item['prob'] > prune_threshold:
                    if edge_item['numeric'] is not None:
                        edge_item['numeric'] = edge_item['numeric'] * learn_rate
                    self[edge] = edge_item

        # copy over edges not learnt
        #
        edges_to_copy = {edge for edge in sub_graph
                         if sub_graph[edge]['edge_type'] not in learn_edge_types}
        for edge in edges_to_copy:
            self[edge] = deepcopy(sub_graph[edge])

        # delete edges with probability close to 0.0
        #
        for edge in edges_to_delete:
            del self[edge]
