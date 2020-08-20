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
                 structure: str = 'star',
                 prune_threshold: float = 0.00001) -> None:
        """
        class that implements the associative memory fabric

        :param uid: str unique name foe this area of the fabric
        :param short_term_memory: the maximum number of neurons in a column of neurons
        :param mp_threshold: the matrix profile window multiplier
        :param structure: str - 'star' structure each column has 4 neighbours  or 'box' structure where each column has 8 neighbours
        :param prune_threshold: float - threshold below which learnt edges are assumed to have zero probability and removed

        """

        # fabric components setup during initialisation
        #
        self.uid: str = uid
        """ the unique name for this neural network """

        self.stm_size: int = short_term_memory
        """ the size of the temporal dimension of the network """

        self.short_term_memory = []
        """ list of last sequence_dimension SDRs received """

        self.mp_threshold = mp_threshold
        """ the matrix profile window multiplier """

        self.structure = structure
        """ the structure of the neuro_columns - star - 5 columns connected in a star, square - 9 neuro_columns connected in a square"""

        self.prune_threshold = prune_threshold
        """ threshold below which learnt edges are assumed to have zero probability and removed """

        self.fabric = NeuralFabric(uid=uid, max_short_term_memory=self.stm_size, mp_threshold=mp_threshold, structure=structure, prune_threshold=prune_threshold)
        """ the fabric of neuro-columns """

        self.non_hebbian_edge_types = set()
        """ the edge_types that are not to be learned using a hebbian rule """

        self.pg = None
        """ the current persist graph """

    def search_for_bmu(self, sdr: SDR, ref_id: Union[str, int], non_hebbian_edges=('generalise',)) -> dict:
        """
        method finds the bmu neuro_column that must be trained and detects if an anomaly or motif has occurred

        :param sdr: sparse data representation of a graph to learn
        :param ref_id: a reference id
        :param non_hebbian_edges: a tuple of edge types identifying edges that will not be learnt using a hebbian rule and not used in the
                                search for the Best Matching Unit
        :return: dict - por
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

        # update the the por
        #
        por['bmu'] = search_results['bmu_coord']
        por['bmu_distance'] = search_results['bmu_distance']
        por['anomaly'] = search_results['anomaly']
        por['motif'] = search_results['motif']
        por['distance_por'] = search_results['fabric_por']
        por['fabric_distance'] = search_results['fabric_distance']

        return por

    def learn(self, search_por: dict) -> dict:

        # prepare the temporal sequence of data by stacking all SDRs in the short_term_memory
        #
        neuro_column = NeuroColumn()
        neuro_column.stack(self.short_term_memory, max_neurons=len(self.short_term_memory))

        hebbian_edges = {neuro_column[edge_key]['edge_type']
                         for edge_key in neuro_column
                         if neuro_column[edge_key]['edge_type'] not in self.non_hebbian_edge_types}

        # assume search_results tuple contains
        # bmu coordinates
        #
        bmu_coord_key = search_por['bmu']

        # bmu distance
        #
        bmu_distance = search_por['bmu_distance']

        # if the bmu distance exceeded the anomaly threshold
        #
        anomaly = search_por['anomaly']

        # the distance to all columns
        #
        fabric_dist = search_por['fabric_distance']

        # if this is an anomaly then want to find the neuron column that is currently on the edge of the fabric
        # any column on the edge cannot have been a BMU as each column keeps track the last time it was the BMU
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
        existing_coords = set(self.fabric.neurons.keys())
        new_coords = set()
        if ((self.fabric.structure == 'star' and len(self.fabric.neurons[bmu_coord_key]['nn']) < 4) or
            (self.fabric.structure == 'box' and len(self.fabric.neurons[bmu_coord_key]['nn']) < 8)):
            self.fabric.grow(example_neuro_column=neuro_column, coord_key=bmu_coord_key, hebbian_edges=hebbian_edges)
            new_coords = set(self.fabric.neurons.keys()) - existing_coords

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

        learn_por = {'updated_bmu': bmu_coord_key,
                     'updated_bmu_learn_rate': bmu_distance,
                     'updated_nn': [nn for nn in coords_to_update if nn != bmu_coord_key],
                     'updated_nn_learn_rate': [learn_rates[idx]
                                               for idx in range(len(coords_to_update))
                                               if coords_to_update[idx] != bmu_coord_key],
                     'coords_grown': new_coords
                     }

        return learn_por

    def train(self, sdr: SDR, ref_id: Union[str, int], non_hebbian_edges=('generalise',)) -> dict:
        """
        method trains the neural network with one NeuroColumn

        :param sdr: sparse data representation of a graph to learn
        :param ref_id: a reference id
        :param non_hebbian_edges: a tuple of edge types identifying edges that will not be learnt using a hebbian rule and not used in the
                                search for the Best Matching Unit
        :return: dict - the Path Of Reasoning
        """
        # search for the bmu and calculate anomalies/ motifs
        #
        search_por = self.search_for_bmu(sdr=sdr, ref_id=ref_id, non_hebbian_edges=non_hebbian_edges)

        # learn the current short term memory given the search_por results
        #
        learn_por = self.learn(search_por=search_por)

        # combine pors
        #
        search_por.update(learn_por)
        return search_por

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
            bmu_coord_key = search_results['bmu_coord']
            result['coords'].append(bmu_coord_key)

            # decode the bmu sdr
            #
            result['neuro_column'] = self.fabric.neurons[bmu_coord_key]['neuro_column'].decode()
        else:

            # create a list of distances and neuron coords, sort in assending order of distance and then pick out the first top_n
            #
            distances = [(coord_key, search_results['fabric_distance'][coord_key]['distance']) for coord_key in search_results['fabric_distance']]
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
        """
        method to return a graph representation (persistence graph) of the fabric
        :param only_updated: If True only the changed data is included
        :return: the persistence graph
        """

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

        edge_properties = {'short_term_memory': [deepcopy(sdr.sdr) for sdr in self.short_term_memory]}
        self.pg.update_edge(source=amfabric_node, target=amfabric_node, edge=('has_short_term_memory', None), **edge_properties)

        edge_properties = {'non_hebbian_edge_types': list(self.non_hebbian_edge_types)}
        self.pg.update_edge(source=amfabric_node, target=amfabric_node, edge=('has_non_hebbian_edge_types', None), **edge_properties)

        # update the edges to each neuro_column
        #
        for coord in fabric['neuro_columns']:
            column_node = ('NeuroColumn', '{}:{}'.format(self.uid, coord))

            # connect fabric to neuro_column with has_neuro_column
            #
            edge_properties = {attr: fabric['neuro_columns'][coord][attr]
                               for attr in fabric['neuro_columns'][coord]
                               if attr not in ['neuro_column']}
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
                                    _generalised_edge_uid=fabric['neuro_columns'][coord]['neuro_column'][edge]['edge_uid'],
                                    _neuro_column=coord,
                                    _coord=fabric['neuro_columns'][coord]['coord'],
                                    _amfabric=self.uid,
                                    _neuron_id=fabric['neuro_columns'][coord]['neuro_column'][edge]['neuron_id'])

                # connect neuro_column to the the generalised node
                #
                if not added_generalised_node and fabric['neuro_columns'][coord]['neuro_column'][edge]['edge_type'] == 'generalise':
                    added_generalised_node = True
                    self.pg.update_edge(source=column_node, target=target_node, edge=('has_generalised', None),
                                        prob=1.0)
        return self.pg

    def set_persist_graph(self, pg: AMFGraph) -> None:
        """
        method to initialise the fabric from a graph representation (the persistence graph)

        :param pg: the persistence graph
        :return: None
        """
        self.pg = pg

        self.short_term_memory = []
        has_stm = self.pg[('AMFabric', self.uid)][('AMFabric', self.uid)][(('has_short_term_memory', None), None)]
        for stm_sdr in has_stm['short_term_memory']:
            sdr = SDR()
            for edge_key in stm_sdr:
                sdr.set_item(source_node=(stm_sdr[edge_key]['source_type'], stm_sdr[edge_key]['source_uid']),
                             target_node=(stm_sdr[edge_key]['target_type'], stm_sdr[edge_key]['target_uid']),
                             edge=(stm_sdr[edge_key]['edge_type'], stm_sdr[edge_key]['edge_uid']),
                             probability=stm_sdr[edge_key]['prob'],
                             numeric=stm_sdr[edge_key]['numeric'],
                             numeric_min=stm_sdr[edge_key]['numeric_min'],
                             numeric_max=stm_sdr[edge_key]['numeric_max'],
                             )
            self.short_term_memory.append(sdr)

        has_non_hebbian_edge_types = self.pg[('AMFabric', self.uid)][('AMFabric', self.uid)][(('has_non_hebbian_edge_types', None), None)]
        self.non_hebbian_edge_types = set(has_non_hebbian_edge_types['non_hebbian_edge_types'])

        has_stats = self.pg[('AMFabric', self.uid)][('AMFabric', self.uid)][(('has_stats', None), None)]
        fabric = {attr: deepcopy(has_stats[attr]) for attr in has_stats if attr[0] != '_'}

        fabric['neuro_columns'] = {}

        for target in self.pg[('AMFabric', self.uid)]:
            if target != ('AMFabric', self.uid):
                for edge in self.pg[('AMFabric', self.uid)][target]:
                    coord = tuple(self.pg[('AMFabric', self.uid)][target][edge]['coord'])

                    fabric['neuro_columns'][coord] = {attr: self.pg[('AMFabric', self.uid)][target][edge][attr]
                                                      for attr in self.pg[('AMFabric', self.uid)][target][edge]
                                                      if attr[0] != '_' and attr not in ['community_nc']}

                    # correct types from lists to sets or tuples
                    #
                    fabric['neuro_columns'][coord]['coord'] = tuple(fabric['neuro_columns'][coord]['coord'])
                    fabric['neuro_columns'][coord]['nn'] = set(fabric['neuro_columns'][coord]['nn'])

                    # place holder for neuro_column
                    #
                    fabric['neuro_columns'][coord]['neuro_column'] = {}

                    # fill out community sdr
                    #
                    fabric['neuro_columns'][coord]['community_nc'] = SDR()
                    for community_edge_key in self.pg[('AMFabric', self.uid)][target][edge]['community_nc']:
                        source_node = (self.pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['source_type'],
                                       self.pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['source_uid'])
                        target_node = (self.pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['target_type'],
                                       self.pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['target_uid'])
                        community_edge = (self.pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['edge_type'],
                                          self.pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['edge_uid'])

                        prob = self.pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['prob']

                        fabric['neuro_columns'][coord]['community_nc'].set_item(source_node=source_node,
                                                                                target_node=target_node,
                                                                                edge=community_edge, probability=prob)
        # loop fills out neuro_columns as stacks of sdrs
        #
        for source in self.pg:
            if source[0] not in ['AMFabric', 'NeuroColumn']:
                for target in self.pg[source]:
                    for edge in self.pg[source][target]:
                        edge_attr = self.pg[source][target][edge]
                        coord = tuple(edge_attr['_coord'])

                        # add sdr for this neuron_id if required
                        #
                        if edge_attr['_neuron_id'] not in fabric['neuro_columns'][coord]['neuro_column']:
                            fabric['neuro_columns'][coord]['neuro_column'][edge_attr['_neuron_id']] = SDR()

                        fabric['neuro_columns'][coord]['neuro_column'][edge_attr['_neuron_id']].set_item(source_node=source,
                                                                                                         target_node=target,
                                                                                                         edge=(edge[0][0], edge_attr['_generalised_edge_uid']),
                                                                                                         probability=edge_attr['_prob'],
                                                                                                         numeric=edge_attr['_numeric'],
                                                                                                         numeric_min=edge_attr['_numeric_min'],
                                                                                                         numeric_max=edge_attr['_numeric_max'],
                                                                                                         )

        self.fabric = NeuralFabric(uid=self.uid,
                                   max_short_term_memory=self.stm_size,
                                   mp_threshold=self.mp_threshold,
                                   structure=self.structure,
                                   prune_threshold=self.prune_threshold)

        self.fabric.restore(fabric)

