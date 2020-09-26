#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Dict, Union, Set, Tuple, List
from src.sdr import SDR, SDRKeyType
import cython
import random


EdgeFeatureType = Union[str, float, int]
""" EdgeFeatureType can be either a string, float or int """

EdgeFeatureKeyType = str
""" EdgeFeatureKeyType are strings """

EdgeKeyType = str

FeatureMapType = Dict[EdgeKeyType, Dict[EdgeFeatureKeyType, EdgeFeatureType]]
""" type to define the structure edges in a column of neurons. Each edge is a string
    and a dictionary holds the edge attributes 
"""

FilterType = Set[str]
""" Filters are sets of strings """


@cython.cclass
class NeuroColumn:

    # help cython declare the instance variables
    #
    edges = cython.declare(dict, visibility='public')
    prune_threshold = cython.declare(cython.double, visibility='public')

    def __init__(self, neuro_column=None, prune_threshold: cython.double = 0.00001) -> None:
        """
        class to implement Spare Data Representation of the features of a sub_graph

        :param neuro_column: a NeuroColumn to copy
        """

        self.edges: FeatureMapType = {}
        """ a dictionary of the edge features"""

        self.prune_threshold = prune_threshold
        """ the threshold below with an edge probability is assumed to be zero and will be deleted """

        # help Cython Static Type
        #
        edge_feature_key: EdgeFeatureKeyType
        edge_key: EdgeKeyType

        if neuro_column is not None:

            # copy over the feature map
            #
            self.edges = {edge_key: {edge_feature_key: neuro_column.edges[edge_key][edge_feature_key]
                                     for edge_feature_key in neuro_column.edges[edge_key]}
                          for edge_key in neuro_column.edges}

    @cython.ccall
    def upsert(self,
               edge_type: str,
               edge_uid: str,
               source_type: str,
               source_uid: str,
               target_type: str,
               target_uid: str,
               neuron_id: cython.int,
               prob: cython.double,
               numeric: float = None,
               numeric_min: float = None,
               numeric_max: float = None):
        """
        method to insert an edge into the SDR

        :param edge_type: the type of edge
        :param edge_uid: the unique name of the edge
        :param source_type: the type of the source node
        :param source_uid: the unique name of the source node
        :param target_type: the type of the target node
        :param target_uid: the name of the target node
        :param neuron_id: number between 0 and max number of neurons on a mini-column
        :param prob: the prob of the edge connection
        :param numeric: the numeric numeric of the edge
        :param numeric_min: the minimum that parameter numeric can be
        :param numeric_max: the maximum that [arameter numeric can be
        :return: None
        """

        # the edge key needs to be unique
        #
        edge_key: EdgeKeyType = '{}:{}:{}:{}:{}:{}:{}'.format(source_type, source_uid, edge_type, edge_uid, neuron_id, target_type, target_uid)

        if edge_key not in self.edges:
            self.edges[edge_key] = {}

        self.edges[edge_key]['source_type'] = source_type
        self.edges[edge_key]['source_uid'] = source_uid
        self.edges[edge_key]['edge_type'] = edge_type
        self.edges[edge_key]['edge_uid'] = edge_uid
        self.edges[edge_key]['neuron_id'] = neuron_id
        self.edges[edge_key]['target_type'] = target_type
        self.edges[edge_key]['target_uid'] = target_uid
        self.edges[edge_key]['prob'] = prob
        # flag to indicate this edge has been changed
        #
        self.edges[edge_key]['updated'] = True

        # add numeric if specified
        #
        if numeric is not None:

            # normalise numeric if min and max provided
            #
            if numeric_min is not None and numeric_max is not None:
                self.edges[edge_key]['numeric'] = (numeric - numeric_min) / (numeric_max - numeric_min)
                self.edges[edge_key]['numeric_min'] = numeric_min
                self.edges[edge_key]['numeric_max'] = numeric_max
            else:
                self.edges[edge_key]['numeric'] = numeric

    def upsert_sdr(self, sdr: SDR, neuron_id: cython.int = 0) -> None:
        """
        method to copy from a sparse data representation of a graph

        :param sdr: the sdr to copy
        :param neuron_id: the neuron_id to assign the sdr edges to
        :return: None
        """
        sdr_key: SDRKeyType

        for sdr_key in sdr:
            self.upsert(edge_type=sdr[sdr_key]['edge_type'],
                        edge_uid=sdr[sdr_key]['edge_uid'],
                        source_type=sdr[sdr_key]['source_type'],
                        source_uid=sdr[sdr_key]['source_uid'],
                        target_type=sdr[sdr_key]['target_type'],
                        target_uid=sdr[sdr_key]['target_uid'],
                        neuron_id=neuron_id,
                        prob=sdr[sdr_key]['prob'],
                        numeric=sdr[sdr_key]['numeric'],
                        numeric_min=sdr[sdr_key]['numeric_min'],
                        numeric_max=sdr[sdr_key]['numeric_max']
                        )

    def stack(self, sdrs: List[SDR], max_neurons: cython.int) -> None:
        """
        method to stack a list of Sparse data representations of graphs together to form a column of neurons
        :param sdrs: the list of SDRs to stack
        :param max_neurons: the maxmimum number of neurons that can be used _ This must be <= length of neuro_column list
        :return: None
        """

        # help cython to static type
        #
        idx: cython.int
        neuron_id: cython.int
        sdr_key: SDRKeyType

        for idx in range(min(len(sdrs), max_neurons)):

            # assume first neuro_column in list will have the largest sequence index
            #
            neuron_id = max_neurons - idx - 1

            for sdr_key in sdrs[idx]:
                self.upsert(edge_type=sdrs[idx][sdr_key]['edge_type'],
                            edge_uid=sdrs[idx][sdr_key]['edge_uid'],
                            source_type=sdrs[idx][sdr_key]['source_type'],
                            source_uid=sdrs[idx][sdr_key]['source_uid'],
                            target_type=sdrs[idx][sdr_key]['target_type'],
                            target_uid=sdrs[idx][sdr_key]['target_uid'],
                            neuron_id=neuron_id,
                            prob=sdrs[idx][sdr_key]['prob'],
                            numeric=sdrs[idx][sdr_key]['numeric'],
                            numeric_min=sdrs[idx][sdr_key]['numeric_min'],
                            numeric_max=sdrs[idx][sdr_key]['numeric_max']
                            )

    def calc_distance_jaccard(self, neuro_column, edge_type_filters: Optional[FilterType] = None, neuron_id_filters: Optional[Set[int]] = None) -> Tuple[float, float, dict]:
        """
        method to calculate the distance between two SDRs
        :param neuro_column: the neuro_column to compare to
        :param edge_type_filters: a set of edge types to compare
        :param neuron_id_filters: a set of neuron_ids to compare
        :return: a tuple of the distance and por - a dictionary keyed by edge
        """

        # help cython static type
        #
        similarity: cython.double
        distance: cython.double
        sum_min: cython.double = 0.0
        sum_max: cython.double = 0.0
        por: dict = {}
        edge_key: EdgeKeyType
        edges_to_process: Set[EdgeKeyType]

        # filter edge_keys as required
        #
        edges_to_process = ({edge_key for edge_key in neuro_column.edges
                            if (edge_type_filters is None or neuro_column.edges[edge_key]['edge_type'] in edge_type_filters) and
                             (neuron_id_filters is None or neuro_column.edges[edge_key]['neuron_id'] in neuron_id_filters)} |
                            {edge_key for edge_key in self.edges
                             if (edge_type_filters is None or self.edges[edge_key]['edge_type'] in edge_type_filters) and
                             (neuron_id_filters is None or self.edges[edge_key]['neuron_id'] in neuron_id_filters)
                             })

        # compare each edge_key
        #
        max_dist = 0.0
        for edge_key in edges_to_process:

            # edge_key in both NeuroColumns
            #
            if edge_key in self.edges and edge_key in neuro_column.edges:

                # por keyed by neuron_id
                #
                if self.edges[edge_key]['neuron_id'] not in por:
                    por[self.edges[edge_key]['neuron_id']] = {'distance': 0.0, 'edges': {}}

                # the distance between probabilities
                #
                edge_dist = abs(self.edges[edge_key]['prob'] - neuro_column.edges[edge_key]['prob'])
                por[self.edges[edge_key]['neuron_id']]['distance'] += edge_dist
                por[self.edges[edge_key]['neuron_id']]['edges'][edge_key] = {'prob': {'distance': edge_dist,
                                                                                      'min': None,
                                                                                      'max': None},
                                                                             'numeric': {'distance': None, 'min': None, 'max': None},
                                                                             'distance': edge_dist
                                                                             }

                if neuro_column.edges[edge_key]['prob'] > self.edges[edge_key]['prob']:
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['prob']['min'] = self.edges[edge_key]['prob']
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['prob']['max'] = neuro_column.edges[edge_key]['prob']

                    sum_min += self.edges[edge_key]['prob']
                    sum_max += neuro_column.edges[edge_key]['prob']
                    max_dist += 1
                else:
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['prob']['max'] = self.edges[edge_key]['prob']
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['prob']['min'] = neuro_column.edges[edge_key]['prob']

                    sum_min += neuro_column.edges[edge_key]['prob']
                    sum_max += self.edges[edge_key]['prob']
                    max_dist += 1

                if 'numeric' in neuro_column.edges[edge_key]:

                    # the distance between numerics
                    #
                    edge_dist = abs(self.edges[edge_key]['numeric'] - neuro_column.edges[edge_key]['numeric'])
                    por[self.edges[edge_key]['neuron_id']]['distance'] += edge_dist
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['distance'] += edge_dist
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['distance'] = edge_dist

                    if neuro_column.edges[edge_key]['numeric'] > self.edges[edge_key]['numeric']:
                        por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['min'] = self.edges[edge_key]['numeric']
                        por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['max'] = neuro_column.edges[edge_key]['numeric']

                        sum_min += self.edges[edge_key]['numeric']
                        sum_max += neuro_column.edges[edge_key]['numeric']
                        max_dist += 1

                    else:
                        por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['max'] = self.edges[edge_key]['numeric']
                        por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['min'] = neuro_column.edges[edge_key]['numeric']

                        sum_min += neuro_column.edges[edge_key]['numeric']
                        sum_max += self.edges[edge_key]['numeric']
                        max_dist += 1

                else:
                    # if no numeric then add place holder to remove numeric bias
                    #
                    sum_min += 1.0
                    sum_max += 1.0
                    max_dist += 1

            # edge key only in this NeuroColumn
            #
            elif edge_key in self.edges:

                # por keyed by neuron_id
                #
                if self.edges[edge_key]['neuron_id'] not in por:
                    por[self.edges[edge_key]['neuron_id']] = {'distance': 0.0, 'edges': {}}

                # the distance between probabilities
                #
                edge_dist = self.edges[edge_key]['prob']
                por[self.edges[edge_key]['neuron_id']]['distance'] += edge_dist
                por[self.edges[edge_key]['neuron_id']]['edges'][edge_key] = {'prob': {'distance': edge_dist,
                                                                                      'min': 0.0,
                                                                                      'max': edge_dist},
                                                                             'numeric': {'distance': None, 'min': None, 'max': None},
                                                                             'distance': edge_dist}
                sum_max += edge_dist
                max_dist += 1

                if 'numeric' in self.edges[edge_key]:

                    edge_dist = self.edges[edge_key]['numeric']
                    por[self.edges[edge_key]['neuron_id']]['distance'] += edge_dist
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['distance'] += edge_dist
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['max'] = edge_dist
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['min'] = 0.0

                    sum_max += edge_dist
                    max_dist += 1

                else:
                    # if no numeric then add place holder to remove numeric bias
                    #
                    sum_min += 1.0
                    sum_max += 1.0
                    max_dist += 1

            # edge_key in the NeuroColumn to compare to
            #
            else:

                # por keyed by neuron_id
                #
                if neuro_column.edges[edge_key]['neuron_id'] not in por:
                    por[neuro_column.edges[edge_key]['neuron_id']] = {'distance': 0.0, 'edges': {}}

                # the distance between probabilities
                #
                edge_dist = neuro_column.edges[edge_key]['prob']
                por[neuro_column.edges[edge_key]['neuron_id']]['distance'] += edge_dist
                por[neuro_column.edges[edge_key]['neuron_id']]['edges'][edge_key] = {'prob': {'distance': edge_dist,
                                                                                              'min': 0.0,
                                                                                              'max': edge_dist},
                                                                                     'numeric': {'distance': None, 'min': None, 'max': None},
                                                                                     'distance': edge_dist
                                                                                     }
                sum_max += edge_dist
                max_dist += 1

                if 'numeric' in neuro_column.edges[edge_key]:

                    # the distance between numeric
                    #
                    edge_dist = neuro_column.edges[edge_key]['numeric']
                    por[neuro_column.edges[edge_key]['neuron_id']]['distance'] += edge_dist
                    por[neuro_column.edges[edge_key]['neuron_id']]['edges'][edge_key]['distance'] += edge_dist
                    por[neuro_column.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['max'] = edge_dist
                    por[neuro_column.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['min'] = 0.0
                    sum_max += edge_dist
                    max_dist += 1

                else:
                    # if no numeric then add place holder to remove numeric bias
                    #
                    sum_min += 1.0
                    sum_max += 1.0
                    max_dist += 1

        similarity = 0.0
        distance = 1.0
        if sum_max > 0:

            # weighted Jaccard similarity is sum of min / sum of max
            #
            #similarity = (sum_min / sum_max)

            # total distance is diff between sum_max and sum_min
            #
            distance = sum_max - sum_min
            similarity = 1 - (distance / max_dist)

        return distance, similarity, por

    def calc_distance(self, neuro_column, edge_type_filters: Optional[FilterType] = None, neuron_id_filters: Optional[Set[int]] = None) -> Tuple[float, float, dict]:
        """
        method to calculate the distance between two SDRs
        :param neuro_column: the neuro_column to compare to
        :param edge_type_filters: a set of edge types to compare
        :param neuron_id_filters: a set of neuron_ids to compare
        :return: a tuple of the distance and por - a dictionary keyed by edge
        """

        # help cython static type
        #
        similarity: cython.double = 0.0
        distance: cython.double = 0.0
        max_dist: cython.double = 0.0
        por: dict = {}
        edge_key: EdgeKeyType
        edges_to_process: Set[EdgeKeyType]

        # filter edge_keys as required
        #
        edges_to_process = ({edge_key for edge_key in neuro_column.edges
                            if (edge_type_filters is None or neuro_column.edges[edge_key]['edge_type'] in edge_type_filters) and
                             (neuron_id_filters is None or neuro_column.edges[edge_key]['neuron_id'] in neuron_id_filters)} |
                            {edge_key for edge_key in self.edges
                             if (edge_type_filters is None or self.edges[edge_key]['edge_type'] in edge_type_filters) and
                             (neuron_id_filters is None or self.edges[edge_key]['neuron_id'] in neuron_id_filters)
                             })

        # compare each edge_key
        #
        for edge_key in edges_to_process:

            # assume every edge has 2 values to be compared
            #
            max_dist += 2.0

            # edge_key in both NeuroColumns
            #
            if edge_key in self.edges and edge_key in neuro_column.edges:

                # por keyed by neuron_id
                #
                if self.edges[edge_key]['neuron_id'] not in por:
                    por[self.edges[edge_key]['neuron_id']] = {'distance': 0.0, 'edges': {}}

                # the distance between probabilities
                #
                edge_dist = abs(self.edges[edge_key]['prob'] - neuro_column.edges[edge_key]['prob'])
                por[self.edges[edge_key]['neuron_id']]['distance'] += edge_dist
                por[self.edges[edge_key]['neuron_id']]['edges'][edge_key] = {'prob': {'distance': edge_dist,
                                                                                      'nc': self.edges[edge_key]['prob'],
                                                                                      'compare_nc': neuro_column.edges[edge_key]['prob']},
                                                                             'numeric': {'distance': 0.0,
                                                                                         'nc': 1.0,
                                                                                         'compare_nc': 1.0},
                                                                             'distance': edge_dist
                                                                             }
                if 'numeric' in neuro_column.edges[edge_key]:

                    # the distance between numerics
                    #
                    edge_dist = abs(self.edges[edge_key]['numeric'] - neuro_column.edges[edge_key]['numeric'])
                    por[self.edges[edge_key]['neuron_id']]['distance'] += edge_dist
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['distance'] += edge_dist
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['distance'] = edge_dist
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['nc'] = self.edges[edge_key]['numeric']
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['compare_nc'] = neuro_column.edges[edge_key]['numeric']

                distance += por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['distance']

            # edge key only in this NeuroColumn
            #
            elif edge_key in self.edges:

                # por keyed by neuron_id
                #
                if self.edges[edge_key]['neuron_id'] not in por:
                    por[self.edges[edge_key]['neuron_id']] = {'distance': 0.0, 'edges': {}}

                # the distance between probabilities
                #
                edge_dist = self.edges[edge_key]['prob']
                por[self.edges[edge_key]['neuron_id']]['distance'] += edge_dist
                por[self.edges[edge_key]['neuron_id']]['edges'][edge_key] = {'prob': {'distance': edge_dist,
                                                                                      'nc': edge_dist,
                                                                                      'compare_nc': 0.0},
                                                                             'numeric': {'distance': 0.0,
                                                                                         'nc': 1.0,
                                                                                         'compare_nc': 1.0},
                                                                             'distance': edge_dist}
                if 'numeric' in self.edges[edge_key]:

                    edge_dist = self.edges[edge_key]['numeric']
                    por[self.edges[edge_key]['neuron_id']]['distance'] += edge_dist
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['distance'] += edge_dist
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['nc'] = edge_dist
                    por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['compare_nc'] = 0.0

                distance += por[self.edges[edge_key]['neuron_id']]['edges'][edge_key]['distance']

            # edge_key in the NeuroColumn to compare to
            #
            else:

                # por keyed by neuron_id
                #
                if neuro_column.edges[edge_key]['neuron_id'] not in por:
                    por[neuro_column.edges[edge_key]['neuron_id']] = {'distance': 0.0, 'edges': {}}

                # the distance between probabilities
                #
                edge_dist = neuro_column.edges[edge_key]['prob']
                por[neuro_column.edges[edge_key]['neuron_id']]['distance'] += edge_dist
                por[neuro_column.edges[edge_key]['neuron_id']]['edges'][edge_key] = {'prob': {'distance': edge_dist,
                                                                                              'nc': 0.0,
                                                                                              'compare_nc': edge_dist},
                                                                                     'numeric': {'distance': 0.0,
                                                                                                 'nc': 1.0,
                                                                                                 'max': 1.0},
                                                                                     'distance': edge_dist
                                                                                     }
                if 'numeric' in neuro_column.edges[edge_key]:

                    # the distance between numeric
                    #
                    edge_dist = neuro_column.edges[edge_key]['numeric']
                    por[neuro_column.edges[edge_key]['neuron_id']]['distance'] += edge_dist
                    por[neuro_column.edges[edge_key]['neuron_id']]['edges'][edge_key]['distance'] += edge_dist
                    por[neuro_column.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['nc'] = 0.0
                    por[neuro_column.edges[edge_key]['neuron_id']]['edges'][edge_key]['numeric']['compare_nc'] = edge_dist

                distance += por[neuro_column.edges[edge_key]['neuron_id']]['edges'][edge_key]['distance']

        if max_dist > 0.0:
            similarity = 1.0 - (distance / max_dist)

        return distance, similarity, por

    def learn(self, neuro_column, learn_rate: cython.double, is_bmu: bool = True, hebbian_edges: Optional[FilterType] = None) -> None:
        """
        method to learn from the specified SDR

        :param neuro_column: the neuron_column to learn from
        :param learn_rate: the hebbian learning rate to apply
        :param is_bmu: if true then will also learn non-hebbian edges
        :param hebbian_edges: a set of edge types to perform hebbian learning on
        :return: None
        """

        # help cython static type
        #
        edge_key: EdgeKeyType
        edges_to_process: Set[EdgeKeyType]
        edges_to_delete: Set[EdgeKeyType]

        # filter edge_keys as required
        #
        edges_to_process = set(self.edges.keys()) | set(neuro_column.edges.keys())

        edges_to_delete = set()
        for edge_key in edges_to_process:

            # edge_key in both self and neuro_column
            #
            if edge_key in self.edges and edge_key in neuro_column.edges:
                if hebbian_edges is None or self.edges[edge_key]['edge_type'] in hebbian_edges:

                    # this edge has been updated
                    #
                    self.edges[edge_key]['updated'] = True

                    # learn new prob and numeric
                    #
                    self.edges[edge_key]['prob'] += (neuro_column.edges[edge_key]['prob'] - self.edges[edge_key]['prob']) * learn_rate
                    if 'numeric' in self.edges[edge_key] and 'numeric' in neuro_column.edges[edge_key]:
                        self.edges[edge_key]['numeric'] += (neuro_column.edges[edge_key]['numeric'] - self.edges[edge_key]['numeric']) * learn_rate

            # edge_key only in self
            #
            elif edge_key in self.edges:

                if hebbian_edges is None or self.edges[edge_key]['edge_type'] in hebbian_edges:
                    # this edge has been updated
                    #
                    self.edges[edge_key]['updated'] = True

                    # learn to forget prob
                    #
                    self.edges[edge_key]['prob'] *= (1.0 - learn_rate)

            # edge_key only in the neuro_column to learn from
            #
            else:
                if hebbian_edges is None or neuro_column.edges[edge_key]['edge_type'] in hebbian_edges:

                    self.edges[edge_key] = {'edge_type': neuro_column.edges[edge_key]['edge_type'],
                                            'edge_uid': neuro_column.edges[edge_key]['edge_uid'],
                                            'source_type': neuro_column.edges[edge_key]['source_type'],
                                            'source_uid': neuro_column.edges[edge_key]['source_uid'],
                                            'target_type': neuro_column.edges[edge_key]['target_type'],
                                            'target_uid': neuro_column.edges[edge_key]['target_uid'],
                                            'neuron_id': neuro_column.edges[edge_key]['neuron_id'],
                                            'prob': neuro_column.edges[edge_key]['prob'] * learn_rate,
                                            'updated': True}

                    # if numeric exists then just copy as it is a new edge in self
                    #
                    if 'numeric' in neuro_column.edges[edge_key]:
                        self.edges[edge_key]['numeric'] = neuro_column.edges[edge_key]['numeric']
                        if 'numeric_min' in neuro_column.edges[edge_key] and 'numeric_max' in neuro_column.edges[edge_key]:
                            self.edges[edge_key]['numeric_min'] = neuro_column.edges[edge_key]['numeric_min']
                            self.edges[edge_key]['numeric_max'] = neuro_column.edges[edge_key]['numeric_max']
                elif is_bmu:

                    self.edges[edge_key] = {'edge_type': neuro_column.edges[edge_key]['edge_type'],
                                            'edge_uid': neuro_column.edges[edge_key]['edge_uid'],
                                            'source_type': neuro_column.edges[edge_key]['source_type'],
                                            'source_uid': neuro_column.edges[edge_key]['source_uid'],
                                            'target_type': neuro_column.edges[edge_key]['target_type'],
                                            'target_uid': neuro_column.edges[edge_key]['target_uid'],
                                            'neuron_id': neuro_column.edges[edge_key]['neuron_id'],
                                            'prob': neuro_column.edges[edge_key]['prob'],
                                            'updated': True}

                    # if numeric exists then just copy as it is a new edge in self
                    #
                    if 'numeric' in neuro_column.edges[edge_key]:
                        self.edges[edge_key]['numeric'] = neuro_column.edges[edge_key]['numeric']
                        if 'numeric_min' in neuro_column.edges[edge_key] and 'numeric_max' in neuro_column.edges[edge_key]:
                            self.edges[edge_key]['numeric_min'] = neuro_column.edges[edge_key]['numeric_min']
                            self.edges[edge_key]['numeric_max'] = neuro_column.edges[edge_key]['numeric_max']

            # add edge to delete list if small enough
            #
            if edge_key in self.edges and self.edges[edge_key]['prob'] < self.prune_threshold:
                edges_to_delete.add(edge_key)

        # delete any edges with close to zero probability
        #
        for edge_key in edges_to_delete:
            del self.edges[edge_key]

    def merge(self, neuro_column, merge_factor: cython.double) -> None:
        """
        method to merge with a NeuroColumn using a merge_factor

        :param neuro_column: the neuro_column to merge with
        :param merge_factor: the merge factor
        :return: None
        """

        # help cython to static type
        #
        edge_key: EdgeKeyType

        # process edges
        #
        edges_to_process: set = set(neuro_column.edges.keys()) | set(self.edges.keys())

        for edge_key in edges_to_process:

            # edge_key in both SDRs
            if edge_key in self.edges and edge_key in neuro_column.edges:

                self.edges[edge_key]['updated'] = True

                self.edges[edge_key]['prob'] += (neuro_column.edges[edge_key]['prob'] * merge_factor)
                if 'numeric' in self.edges[edge_key] and 'numeric' in neuro_column.edges[edge_key]:
                    self.edges[edge_key]['numeric'] += (neuro_column.edges[edge_key]['numeric'] * merge_factor)

            # edge_key only in the SDR to merge with
            #
            elif edge_key in neuro_column.edges:
                self.edges[edge_key] = {'edge_type': neuro_column.edges[edge_key]['edge_type'],
                                        'edge_uid': neuro_column.edges[edge_key]['edge_uid'],
                                        'source_type': neuro_column.edges[edge_key]['source_type'],
                                        'source_uid': neuro_column.edges[edge_key]['source_uid'],
                                        'target_type': neuro_column.edges[edge_key]['target_type'],
                                        'target_uid': neuro_column.edges[edge_key]['target_uid'],
                                        'neuron_id': neuro_column.edges[edge_key]['neuron_id'],
                                        'prob': neuro_column.edges[edge_key]['prob'] * merge_factor,
                                        'updated': True}

                if 'numeric' in neuro_column.edges[edge_key]:
                    self.edges[edge_key]['numeric'] = (neuro_column.edges[edge_key]['numeric'] * merge_factor)
                    if 'numeric_min' in neuro_column.edges[edge_key] and 'numeric_max' in neuro_column.edges[edge_key]:
                        self.edges[edge_key]['numeric_min'] = neuro_column.edges[edge_key]['numeric_min']
                        self.edges[edge_key]['numeric_max'] = neuro_column.edges[edge_key]['numeric_max']

    def randomize(self, neuro_column, edges_to_randomise: set = None) -> None:
        """
        method to randomise this SDR based on the example SDR
        :param neuro_column: the example SDR
        :param edges_to_randomise: optional set of edges to randomise
        :return: None
        """

        # help cython to static type
        #
        edge_key: EdgeKeyType

        for edge_key in neuro_column.edges:
            if edges_to_randomise is None or neuro_column.edges[edge_key]['edge_type'] in edges_to_randomise:

                rnd_numeric = None
                numeric_min = None
                numeric_max = None

                # calculate a random numeric if required
                #
                if 'numeric' in neuro_column.edges[edge_key]:

                    # use the normalisation boundaries to calc a random number - which will be normalised when upserted...
                    #
                    if 'numeric_min' in neuro_column.edges[edge_key] and 'numeric_max' in neuro_column.edges[edge_key]:
                        rnd_numeric = (random.random() * (neuro_column.edges[edge_key]['numeric_max'] - neuro_column.edges[edge_key]['numeric_min'])) + neuro_column.edges[edge_key]['numeric_min']
                        numeric_min = neuro_column.edges[edge_key]['numeric_min']
                        numeric_max = neuro_column.edges[edge_key]['numeric_max']
                    else:
                        rnd_numeric = random.random()

                self.upsert(edge_type=neuro_column.edges[edge_key]['edge_type'],
                            edge_uid=neuro_column.edges[edge_key]['edge_uid'],
                            source_type=neuro_column.edges[edge_key]['source_type'],
                            source_uid=neuro_column.edges[edge_key]['source_uid'],
                            target_type=neuro_column.edges[edge_key]['target_type'],
                            target_uid=neuro_column.edges[edge_key]['target_uid'],
                            neuron_id=neuro_column.edges[edge_key]['neuron_id'],

                            # Randomise probability
                            #
                            prob=random.random(),
                            numeric=rnd_numeric,
                            numeric_min=numeric_min,
                            numeric_max=numeric_max
                            )

    def get_edge_by_max_probability(self) -> Optional[Dict[EdgeFeatureKeyType, EdgeFeatureType]]:
        """
        method to return the edge with the maximum prob
        :return: Dictionary with keys: 'edge_type', 'edge_uid', 'source_type', 'source_uid', 'target_type', 'target_uid', 'neuron_id', 'prob', Optional['numeric', 'numeric_min', 'numeric_max']

        """

        # help Cython to static type
        #
        max_prob: cython.double = 0.0
        max_edge_key: Optional[EdgeKeyType] = None
        edge_key: EdgeKeyType

        for edge_key in self.edges:
            if max_edge_key is None or self.edges[edge_key]['prob'] >= max_prob:
                max_edge_key = edge_key
                max_prob = self.edges[edge_key]['prob']

        return self.edges[max_edge_key]

    def __str__(self):
        """
        method to display the NeuroColumn as a string
        :return: a string representation of the SDR attributes
        """

        # help cython static type
        #
        txt: str = ''
        edge_key: EdgeKeyType

        for edge_key in self.edges:

            if 'numeric' in self.edges[edge_key]:
                if 'numeric_min' in self.edges[edge_key]:
                    numeric = (self.edges[edge_key]['numeric'] * (self.edges[edge_key]['numeric_min'] - self.edges[edge_key]['numeric_max'])) + self.edges[edge_key]['numeric_min']
                else:
                    numeric = self.edges[edge_key]['numeric']
            else:
                numeric = None
            txt = '{}Sce: {}:{}\nEdge: {}:{}:{}\nTrg: {}:{}\nProb: {}\n'.format(txt,
                                                                                self.edges[edge_key]['source_type'], self.edges[edge_key]['source_uid'],
                                                                                self.edges[edge_key]['edge_type'], self.edges[edge_key]['edge_uid'],
                                                                                self.edges[edge_key]['neuron_id'],
                                                                                self.edges[edge_key]['target_type'], self.edges[edge_key]['target_uid'],
                                                                                self.edges[edge_key]['prob'])
            if numeric is not None:
                txt = '{}Numeric: {}\n'.format(txt, numeric)

        return txt

    def __contains__(self, edge_key: EdgeKeyType) -> bool:
        """
        method to check if an edge_key exists in the NeuroColumn
        :param edge_key: edge key to check
        :return: True if it exists else False
        """

        return edge_key in self.edges

    def __iter__(self) -> iter:
        """
        method to return an iterable of the NeuroColumn edge keys
        :return: iterable of neuro_column keys
        """
        return iter(self.edges)

    def __getitem__(self, edge_key: EdgeKeyType) -> Dict[EdgeFeatureKeyType, EdgeFeatureType]:
        """
        method to access the NeuroColumn edge attributes
        :param edge_key: the edge to return
        :return: the edge attributes
        """
        return self.edges[edge_key]

    def decode(self, only_updated: bool = False) -> FeatureMapType:
        """
        method to return a dictionary representation of the NeuroColumn. Any normalised numeric will be denormalised

        :param only_updated: Set to true to return on updated edges
        :return: dictionary of dictionaries with outer dict keyed by edge_key, inner dictionary with keys:\n
                    'edge_type', 'edge_uid', 'source_type', 'source_name', 'target_type', 'target_name', 'neuron_id', 'prob', Optional['numeric', 'numeric_min', 'numeric_max'] keyed by each edge
        """
        neuro_column: FeatureMapType = {}
        edge_key: EdgeKeyType
        feature_key: EdgeFeatureKeyType

        for edge_key in self.edges:
            if not only_updated or self.edges[edge_key]['updated']:

                neuro_column[edge_key] = {feature_key: self.edges[edge_key][feature_key]
                                          for feature_key in self.edges[edge_key]
                                          if feature_key != 'updated'}

                # denormalise numeric if required
                #
                if 'numeric_min' in neuro_column[edge_key] and 'numeric_max' in neuro_column[edge_key] and 'numeric' in neuro_column[edge_key]:
                    neuro_column[edge_key]['numeric'] = ((neuro_column[edge_key]['numeric'] * (neuro_column[edge_key]['numeric_max'] - neuro_column[edge_key]['numeric_min'])) +
                                                         neuro_column[edge_key]['numeric_min'])

                # reset the update flag if True
                #
                if only_updated:
                    self.edges[edge_key]['updated'] = False

        return neuro_column
