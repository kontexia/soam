#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Dict, Tuple, Union
from copy import deepcopy

# type Aliases
#
SDRNodeType = Tuple[str, str]
""" SDRNodeType is a tuple of strings (node_type and node_uid) """

SDREdgeType = str
""" SDREdgeType is a string edge_type """

SDRKeyType = str
""" SDRKeyType is a tuple of (source_node, edge_type, target_node) """

SDRValueType = Dict[str, Union[str, float, None]]
""" SDRValueType is a dictionary: {prob: float, numeric: float} """


class SDR:

    def __init__(self, sdr=None):
        self.sdr: dict = {}

        if sdr is not None:
            self.sdr = deepcopy(sdr.sdr)

    def __contains__(self, sdr_key: SDRKeyType) -> bool:
        """
        method to check if an sdr_key exists in the sdr
        :param sdr_key: edge key to check
        :return: True if it exists else False
        """

        return sdr_key in self.sdr

    def __iter__(self) -> iter:
        """
        method to return an iterable of the sdr keys
        :return: iterable of self keys
        """
        return iter(self.sdr)

    def __getitem__(self, sdr_key: SDRKeyType) -> SDRValueType:
        """
        method to access the sdr edge attributes
        :param sdr_key: the edge to return
        :return: the edge attributes {source_type: , source_uid:, target_type: , target_uid:, prob:, numeric:, numeric_min:, numeric_max:}
        """
        return self.sdr[sdr_key]

    def set_item(self,
                 source_node: SDRNodeType,
                 edge: SDREdgeType,
                 target_node: SDRNodeType,
                 probability: float = 1.0,
                 numeric: Optional[float] = None,
                 numeric_min: Optional[float] = None,
                 numeric_max: Optional[float] = None) -> None:
        """
        method to set the sdr attributes

        :param source_node: tuple of (source_type, source uid)
        :param edge: edge_type
        :param target_node: tuple of (target_type, target uid)
        :param probability: probability of the edge
        :param numeric: numeric value associated with edge
        :param numeric_min: the min numeric can be
        :param numeric_max: the max numeric can be
        :return: None
        """
        sdr_key = '{}:{}:{}:{}:{}'.format(source_node[0], source_node[1], edge, target_node[0], target_node[1])
        self.sdr[sdr_key] = {'source_type': source_node[0], 'source_uid': source_node[1],
                             'target_type': target_node[0], 'target_uid': target_node[1],
                             'edge_type': edge,
                             'prob': probability,
                             'numeric': numeric,
                             'numeric_min': numeric_min,
                             'numeric_max': numeric_max
                             }

    def update(self, sdr) -> None:
        """
        method to update and sdr
        :param sdr: the sdr to update with
        :return: None
        """
        self.sdr.update(sdr.sdr)
