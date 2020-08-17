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

ContributionType = Dict[str, Union[str, float]]
""" the contribution dictionary specify the edge, prob and numeric contributions"""


@cython.cclass
class NeuroColumn:

    # help cython declare the instance variables
    #
    edges = cython.declare(dict, visibility='public')
    max_neurons = cython.declare(cython.int, visibility='public')

    def __init__(self, neuro_column=None) -> None:
        """
        class to implement Spare Data Representation of the features of a sub_graph

        :param neuro_column: a NeuroColumn to copy
        """

        self.edges: FeatureMapType = {}
        """ a dictionary of the edge features"""

        self.max_neurons = 0
        """ the maximum number of Neurons in NeuroColumn"""

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

            # copy over the maximum sequence
            #
            self.max_neurons = neuro_column.max_neurons

    @cython.ccall
    def upsert(self,
               edge_type: str,
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
        edge_key: EdgeKeyType = '{}:{}:{}:{}:{}:{}'.format(source_type, source_uid, edge_type, neuron_id, target_type, target_uid)

        if edge_key not in self.edges:
            self.edges[edge_key] = {}

        self.edges[edge_key]['source_type'] = source_type
        self.edges[edge_key]['source_uid'] = source_uid
        self.edges[edge_key]['edge_type'] = edge_type
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
            if numeric_min is not None and numeric_min is not None:
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
                        source_type=sdr[sdr_key]['source_type'],
                        source_uid=sdr[sdr_key]['source_uid'],
                        target_type=sdr[sdr_key]['target_type'],
                        target_uid=sdr[sdr_key]['target_uid'],
                        neuron_id=neuron_id,
                        prob=sdr[sdr_key]['prob'],
                        numeric=sdr[sdr_key]['numeric'],
                        numric_min=sdr[sdr_key]['numeric_min'],
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

    def calc_distance(self, neuro_column, filter_types: Optional[FilterType] = None) -> Tuple[float, List[ContributionType]]:
        """
        method to calculate the distance between two SDRs
        :param neuro_column: the neuro_column to compare to
        :param filter_types: a set of edge types to compare
        :return: a tuple of the distance and list of distance contributions {edge: EdgeKeyType, prob: Double, numeric: Double}
        """

        # help cython static type
        #
        distance: cython.double
        sum_min: cython.double = 0.0
        sum_max: cython.double = 0.0
        contributions: List[ContributionType] = []
        edge_key: EdgeKeyType
        edges_to_process: Set[EdgeKeyType]

        # filter edge_keys as required
        #
        edges_to_process = ({edge_key for edge_key in neuro_column.edges
                            if (filter_types is None or neuro_column.edges[edge_key]['edge_type'] in filter_types)} |
                            {edge_key for edge_key in self.edges
                             if (filter_types is None or self.edges[edge_key]['edge_type'] in filter_types)})

        # compare each edge_key
        #
        for edge_key in edges_to_process:

            # edge_key in both NeuroColumns
            #
            if edge_key in self.edges and edge_key in neuro_column.edges:
                sum_min += min(neuro_column.edges[edge_key]['prob'], self.edges[edge_key]['prob'])
                sum_max += max(neuro_column.edges[edge_key]['prob'], self.edges[edge_key]['prob'])
                contributions.append({'edge': edge_key, 'prob': abs(neuro_column.edges[edge_key]['prob'] - self.edges[edge_key]['prob'])})

                if 'numeric' in neuro_column.edges[edge_key]:
                    sum_min += min(neuro_column.edges[edge_key]['numeric'], self.edges[edge_key]['numeric'])
                    sum_max += max(neuro_column.edges[edge_key]['numeric'], self.edges[edge_key]['numeric'])
                    contributions.append({'edge': edge_key, 'numeric': abs(neuro_column.edges[edge_key]['numeric'] - self.edges[edge_key]['numeric'])})

            # edge key only in this NeuroColumn
            #
            elif edge_key in self.edges:
                sum_max += self.edges[edge_key]['prob']
                contributions.append({'edge': edge_key, 'prob': self.edges[edge_key]['prob']})

                if 'numeric' in self.edges[edge_key]:
                    sum_max += self.edges[edge_key]['numeric']
                    contributions.append({'edge': edge_key, 'numeric': self.edges[edge_key]['numeric']})

            # edge_key in the NeuroColumn to compare to
            #
            else:
                sum_max += neuro_column.edges[edge_key]['prob']
                contributions.append({'edge': edge_key, 'prob': neuro_column.edges[edge_key]['prob']})

                if 'numeric' in neuro_column.edges[edge_key]:
                    sum_max += neuro_column.edges[edge_key]['numeric']
                    contributions.append({'edge': edge_key, 'numeric': neuro_column.edges[edge_key]['numeric']})

        distance = 1.0
        if sum_max > 0:

            # weighted Jaccard Distance is 1 - (ratio of sum of mins / sum of maxs)
            #
            distance = 1 - (sum_min / sum_max)

        return distance, contributions

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
        edge_to_process: Set[EdgeKeyType]

        # filter edge_keys as required
        #
        edges_to_process = set(self.edges.keys()) | set(neuro_column.edges.keys())

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
        :return: Dictionary with keys: 'edge_type', 'source_type', 'source_uid', 'target_type', 'target_uid', 'neuron_id', 'prob', Optional['numeric', 'numeric_min', 'numeric_max']

        """

        # help Cython to static type
        #
        max_prob: cython.double = 0.0
        max_edge_key: Optional[EdgeKeyType] = None
        edge_key: EdgeKeyType

        for edge_key in self.edges:
            if self.edges[edge_key]['prob'] > max_prob:
                max_edge_key = edge_key
                max_prob = self.edges[edge_key]['prob']

        if max_edge_key is not None:
            return self.edges[max_edge_key]
        else:
            return None

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
            txt = '{}Sce: {}:{}\nEdge: {}:{}\nTrg: {}:{}\nProb: {}\n'.format(txt,
                                                                             self.edges[edge_key]['source_type'], self.edges[edge_key]['source_uid'],
                                                                             self.edges[edge_key]['edge_type'], self.edges[edge_key]['neuron_id'],
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
                    'edge_type', 'source_type', 'source_name', 'target_type', 'target_name', 'neuron_id', 'prob', Optional['numeric', 'numeric_min', 'numeric_max'] keyed by each edge
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
