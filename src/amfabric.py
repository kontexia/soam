#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy
from src.sdr import SDR
from src.neuro_column import NeuroColumn

from src.neural_fabric import NeuralFabric
from src.amgraph import AMFGraph
from typing import Union, List, Dict


class AMFabric:

    def __init__(self,
                 uid: str,
                 short_term_memory: int = 1,
                 mp_threshold: int = 5,
                 structure: str = 'star') -> None:
        """
        class that implements the associative memory fabric

        :param uid: str unique name foe this area of the fabric
        :param short_term_memory: the maximum number of neurons in a column of neurons
        :param mp_threshold: the matrix profile window multiplier
        :param structure: str - 'star' structure each column has 4 neighbours  or 'box' structure where each column has 8 neighbours
        """

        # fabric components setup during initialisation
        #
        self.uid: str = uid
        """ the unique name for this neural network """

        self.stm_size: int = short_term_memory
        """ the size of the temporal dimension of the network """

        self.short_term_memory = []
        """ list of last sequence_dimension SDRs received """

        self.fabric = NeuralFabric(uid=uid, max_short_term_memory=self.stm_size, mp_threshold=mp_threshold, structure=structure)
        """ the fabric of neuro-columns """

        self.non_hebbian_edge_types = set()
        """ the edge_types that are not to be learned using a hebbian rule """

        self.pg = None
        """ the current persist graph """

    def train(self, sdr: SDR, ref_id: Union[str, int], non_hebbian_edges=('generalise',)) -> dict:
        """
        method trains the neural network with one NeuroColumn

        :param sdr: sparse data representation of a graph to learn
        :param ref_id: a reference id
        :param non_hebbian_edges: a tuple of edge types identifying edges that will not be learnt using a hebbian rule and not used in the
                                search for the Best Matching Unit
        :return: dict - the Path Of Reasoning
        """
        # ref ids have to be converted to strings
        #
        if not isinstance(ref_id, str):
            str_ref_id = str(ref_id)
        else:
            str_ref_id = ref_id

        # start building the por
        #
        por = {'ref_id': str_ref_id,

               # the current anomaly threshold before any update
               #
               'anomaly_threshold': self.fabric.anomaly_threshold,

               # the current motif threshold before any update
               #
               'motif_threshold': self.fabric.motif_threshold,
               }

        # update short term memory and make sure the window size doesnt grow beyond the required boundary
        #
        self.short_term_memory.append(sdr)
        if len(self.short_term_memory) > self.stm_size:
            self.short_term_memory.pop(0)

        # prepare the temporal sequence of data by stacking all SDRs in the short_term_memory
        #
        neuro_column = NeuroColumn()
        neuro_column.stack(self.short_term_memory, max_neurons=len(self.short_term_memory))

        # work out the edge_types for hebbian learning
        #
        if non_hebbian_edges is not None:
            self.non_hebbian_edge_types.update(non_hebbian_edges)

            hebbian_edges = {neuro_column[edge_key]['edge_type']
                             for edge_key in neuro_column
                             if neuro_column[edge_key]['edge_type'] not in self.non_hebbian_edge_types}
        else:
            hebbian_edges = None

        # if the fabric is empty initialise
        #
        if len(self.fabric.neurons) == 0:
            self.fabric.grow(example_neuro_column=neuro_column, hebbian_edges=hebbian_edges)

        # find the BMU by calculating the distance of the NeuroColumn to every column in the fabric
        #
        search_results = self.fabric.distance_to_fabric(neuro_column=neuro_column, ref_id=str_ref_id, bmu_search_filters=hebbian_edges)

        # assume search_results tuple contains
        # bmu coordinates
        #
        bmu_coord_key = search_results[0]

        # bmu distance
        #
        bmu_distance = search_results[1]

        # if the bmu distance exceeded the anomaly threshold
        #
        anomaly = search_results[2]

        # the distance to all columns
        #
        fabric_dist = search_results[4]

        # if this is an anomaly then want to find the neuron column that is currently on the edge of the fabric
        # any column on the edge cannot have been a BMU as each column kkeps track the last time it was the BMU
        # by calculating (mapped - last_bmu) edge columns will have a large number
        # sorting in descending order of (1-distance) * (mapped - last_bmu) will present the closest edge columns
        #

        if anomaly:
            dist_list = [(coord_key,
                          fabric_dist[coord_key]['distance'],
                          (1 - fabric_dist[coord_key]['distance']) * (self.fabric.mapped - fabric_dist[coord_key]['last_bmu']))
                         for coord_key in fabric_dist]
            dist_list.sort(key=lambda x: x[2], reverse=True)
            bmu_coord_key = dist_list[0][0]
            bmu_distance = dist_list[0][1]

        # grow the mini column neighbours if required - note any edge columns will always require new neighbours
        # if selected as the bmu
        #
        if ((self.fabric.structure == 'star' and len(self.fabric.neurons[bmu_coord_key]['nn']) < 4) or
            (self.fabric.structure == 'box' and len(self.fabric.neurons[bmu_coord_key]['nn']) < 8)):
            self.fabric.grow(example_neuro_column=neuro_column, coord_key=bmu_coord_key, hebbian_edges=hebbian_edges)

        # update the bmu and its neighbours stats
        #
        self.fabric.update_bmu_stats(bmu_coord_key=bmu_coord_key, distance=bmu_distance)

        # get neighbourhood of neurons to learn
        #
        coords_to_update = list(self.fabric.neurons[bmu_coord_key]['nn'])

        # add in the bmu coord
        #
        coords_to_update.append(bmu_coord_key)

        # and the learn rates for each mini column of neurons. if mini-column has just been grown then won't be in fabric_dist so default to bmu dist
        #
        learn_rates = [fabric_dist[coord_key]['distance'] if coord_key in fabric_dist else bmu_distance for coord_key in coords_to_update]

        self.fabric.learn(neuro_column=neuro_column, bmu_coord=bmu_coord_key, coords=coords_to_update, learn_rates=learn_rates, hebbian_edges=hebbian_edges)

        # update communities
        #
        self.fabric.community_update(bmu_coord_key=bmu_coord_key, learn_rate=(1 - bmu_distance))

        # update the the por
        #
        por['bmu'] = bmu_coord_key
        por['bmu_distance'] = bmu_distance
        por['anomaly'] = anomaly
        por['motif'] = search_results[3]
        por['distance_por'] = search_results[5]

        return por

    def query(self, sdr, top_n: int = 1) -> Dict[str, Union[List[str], NeuroColumn]]:
        """
        method to query the associative memory neural network
        :param sdr: Is a sdr or list of SDRs that will be used to query the network. If a list it is assumed the order provided sequence context
                    where list[0] is oldest and list[n] is most recent. Also assumes list is not bigger than preconfigure sequence_dimension
        :param top_n: int - if > 1 then the top n neuro-columns will be merged weighted by the closeness to the query SDR(s)
        :return: dict with keys:\n
                        'coords': a list of neuro_column coordinates selected as the best matching units
                        'neuro_column': the BMU(s) neuro_column
        """

        query_nc = NeuroColumn()

        if isinstance(sdr, list):
            query_nc.stack(sdrs=sdr, max_neurons=len(sdr) + 1)
        else:
            query_nc.stack(sdrs=[sdr], max_neurons=1)

        # search only for the edges in the query
        #
        search_edges = {query_nc[edge_key]['edge_type'] for edge_key in query_nc}

        # search the fabric
        #
        search_results = self.fabric.distance_to_fabric(neuro_column=query_nc, bmu_search_filters=search_edges)

        # prepare the result to return
        #
        result = {'coords': [],
                  'neuro_column': None}

        if top_n == 1:
            # assume search_results[0] is the bmu coordinates
            #
            bmu_coord_key = search_results[0]
            result['coords'].append(bmu_coord_key)

            # decode the bmu sdr
            #
            result['neuro_column'] = self.fabric.neurons[bmu_coord_key]['neuro_column'].decode()
        else:

            # create a list of distances and neuron coords, sort in assending order of distance and then pick out the first top_n
            #
            distances = [(coord_key, search_results[4][coord_key]['distance']) for coord_key in search_results[4]]
            distances.sort(key=lambda x: x[1])
            merge_factors = []
            coords_to_merge = []
            n = 0
            idx = 0
            while n < top_n and idx < len(distances):
                if distances[idx][1] < 1.0:
                    coords_to_merge.append(distances[idx][0])
                    merge_factors.append(1 - distances[idx][1])
                    n += 1
                    idx += 1
                else:
                    break

            # merge the top_n neurons with a merge factor equal to  similarity ie 1 - distance
            #
            merged_neuron = self.fabric.merge_neurons(coords=coords_to_merge, merge_factors=merge_factors)
            result['coords'] = coords_to_merge
            result['neuro_column'] = merged_neuron.decode()

        return result

    def decode_fabric(self, all_details: bool = False, only_updated: bool=False, community_sdr: bool = False) -> dict:
        """
        method extracts the neural fabric into a dict structure whilst decoding each SDR

        :param bmu_only: if True will only return neurons that have been a BMU
        :param community_sdr: If True will include the community SDR learnt for each neuron column
        :return: dictionary keyed by neuron column coordinates containing all the attributes
        """
        fabric = self.fabric.decode(all_details=all_details,
                                    community_sdr=community_sdr,
                                    only_updated=only_updated)
        return fabric

    def get_anomalies(self) -> dict:
        """
        method returns a copy of the fabric anomalies, keyed the str(ref_id)
        :return: dict of dict with structure {'bmu_coord':, 'distance':, 'threshold':, 'por':}
        """
        return deepcopy(self.fabric.anomaly)

    def get_motifs(self) -> dict:
        """
        method returns a copy of the fabric morifs, keyed the str(ref_id)
        :return: dict of dict with structure {'bmu_coord':, 'distance':, 'threshold':, 'por':}
        """
        return deepcopy(self.fabric.motif)

    def get_persist_graph(self, only_updated=True):

        fabric = self.fabric.decode(all_details=True,
                                    community_sdr=True,
                                    only_updated=only_updated)

        if self.pg is None:
            self.pg = AMFGraph()

        amfabric_node = ('AMFabric', self.uid)
        if amfabric_node not in self.pg:
            self.pg.set_node(node=amfabric_node)

        # update the statistics for the fabric with self referencing edge has_stats
        #
        edge_properties = {attr: fabric[attr] for attr in fabric if attr not in ['neuro_columns']}
        self.pg.update_edge(source=amfabric_node, target=amfabric_node, edge=('has_stats', None), **edge_properties)

        # update the edges to each neuro_column
        #
        for coord in fabric['neuro_columns']:
            column_node = ('NeuroColumn', '{}:{}'.format(self.uid, coord))

            # connect fabric to neuro_column with has_neuro_column
            #
            edge_properties = {attr: fabric['neuro_columns'][coord][attr] for attr in fabric['neuro_columns'][coord] if attr not in ['neuro_column']}
            self.pg.update_edge(source=amfabric_node, target=column_node, edge=('has_neuro_column', None), **edge_properties)

            added_generalised_node = False
            for edge in fabric['neuro_columns'][coord]['neuro_column']:

                # the name of the edge has to be qualified by the fabric neuro_column
                #
                edge_uid = '{}_{}_{}_{}'.format(self.uid,
                                                fabric['neuro_columns'][coord]['coord'][0],
                                                fabric['neuro_columns'][coord]['coord'][1],
                                                fabric['neuro_columns'][coord]['neuro_column'][edge]['neuron_id']
                                                )
                source_node = (fabric['neuro_columns'][coord]['neuro_column'][edge]['source_type'], fabric['neuro_columns'][coord]['neuro_column'][edge]['source_uid'])
                target_node = (fabric['neuro_columns'][coord]['neuro_column'][edge]['target_type'], fabric['neuro_columns'][coord]['neuro_column'][edge]['target_uid'])
                if 'prob' in fabric['neuro_columns'][coord]['neuro_column'][edge]:
                    prob = fabric['neuro_columns'][coord]['neuro_column'][edge]['prob']
                else:
                    prob = None

                if 'numeric' in fabric['neuro_columns'][coord]['neuro_column'][edge]:
                    numeric = fabric['neuro_columns'][coord]['neuro_column'][edge]['numeric']
                    numeric_min = fabric['neuro_columns'][coord]['neuro_column'][edge]['numeric_min']
                    numeric_max = fabric['neuro_columns'][coord]['neuro_column'][edge]['numeric_max']
                else:
                    numeric = None
                    numeric_min = None
                    numeric_max = None

                edge_key = (fabric['neuro_columns'][coord]['neuro_column'][edge]['edge_type'], edge_uid)

                self.pg.update_edge(source=source_node, target=target_node, edge=edge_key,
                                    prob=prob, numeric=numeric, numeric_min=numeric_min, numeric_max=numeric_max,
                                    _neuro_column=coord,
                                    _amfabric=self.uid,
                                    _neuron_id=fabric['neuro_columns'][coord]['neuro_column'][edge]['neuron_id'])

                # connect neuro_column to the the generalised node
                #
                if not added_generalised_node and fabric['neuro_columns'][coord]['neuro_column'][edge]['edge_type'] == 'generalise':
                    added_generalised_node = True
                    self.pg.update_edge(source=column_node, target=target_node, edge=('has_generalised', None),
                                        prob=1.0)
        return self.pg

