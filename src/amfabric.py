#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy
from src.sdr import SDR
from src.neuro_column import NeuroColumn

from src.neural_fabric import NeuralFabric
from src.amgraph import AMFGraph
from typing import Union, List, Dict, Optional
import random


class AMFabric:

    def __init__(self,
                 uid: str,
                 short_term_memory: int = 1,
                 mp_threshold: int = 5,
                 structure: str = 'star',
                 prune_threshold: float = 0.00001,
                 random_seed=None) -> None:
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

        self.non_hebbian_edge_types = {'edges': set(), 'updated': True}
        """ the edge_types that are not to be learned using a hebbian rule """

        # seed the random number generator if required
        #
        if random_seed is not None:
            random.seed(random_seed)

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
            non_hebbian_edges_set = set(non_hebbian_edges)
            if len(self.non_hebbian_edge_types['edges'] & non_hebbian_edges_set) != len(self.non_hebbian_edge_types['edges'] | non_hebbian_edges_set):
                self.non_hebbian_edge_types['edges'].update(non_hebbian_edges)
                self.non_hebbian_edge_types['updated'] = True

            hebbian_edges = {neuro_column[edge_key]['edge_type']
                             for edge_key in neuro_column
                             if neuro_column[edge_key]['edge_type'] not in self.non_hebbian_edge_types['edges']}
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
        por['fabric_distance'] = search_results['fabric_distance']

        return por

    def learn(self, search_por: dict) -> dict:
        """
        method that learns the current short term memory taking into account the bmu suggested in search_por
        :param search_por: the por resulting from a previous call to search_for_bmu
        :return: learn_por
        """

        # prepare the temporal sequence of data by stacking all SDRs in the short_term_memory
        #
        neuro_column = NeuroColumn()
        neuro_column.stack(self.short_term_memory, max_neurons=len(self.short_term_memory))

        hebbian_edges = {neuro_column[edge_key]['edge_type']
                         for edge_key in neuro_column
                         if neuro_column[edge_key]['edge_type'] not in self.non_hebbian_edge_types['edges']}

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
                     'coords_grown': list(new_coords)
                     }

        return learn_por

    def train(self, sdr: SDR, ref_id: Union[str, int], non_hebbian_edges=('generalise',)) -> dict:
        """
        method trains the neural network with one sdr

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

    def decode_fabric(self, all_details: bool = False, community_sdr: bool = False) -> dict:
        """
        method extracts the neural fabric into a dict structure whilst decoding each SDR

        :param all_details: if True will include all details of the fabric
        :param community_sdr: If True will include the community SDR learnt for each neuron column
        :return: dictionary keyed by neuron column coordinates containing all the attributes
        """
        fabric = self.fabric.decode(all_details=all_details,
                                    community_sdr=community_sdr,
                                    only_updated=False,
                                    reset_updated=False)
        return fabric

    def get_anomalies(self) -> dict:
        """
        method returns a copy of the fabric anomalies, keyed by ref_id
        :return: dict of dict with structure {ref_id: {'bmu_coord':, 'distance':, 'threshold':, 'por':})
        """
        return deepcopy(self.fabric.anomaly)

    def get_motifs(self) -> dict:
        """
        method returns a copy of the fabric motifs, keyed by ref_id
        :return: dict of dict with structure {ref_id: {'bmu_coord':, 'distance':, 'threshold':, 'por':}}
        """
        return deepcopy(self.fabric.motif)

    def get_persist_graph(self, ref_id: Union[str, int], pg_to_update: Optional[AMFGraph] = None, only_updated: bool = True):
        """
        method to return a graph representation (persistence graph) of the fabric

        :param ref_id: a unique reference that will be tagged on every changed edge
        :param pg_to_update: If provided this amfgraph will be updated
        :param only_updated: If True only the changed data is included
        :return: the persistence graph
        """

        # ref ids have to be converted to strings
        #
        if not isinstance(ref_id, str):
            str_ref_id = str(ref_id)
        else:
            str_ref_id = ref_id

        # we will either update existing or create a new graph
        #
        if pg_to_update is None:
            pg = AMFGraph()
        else:
            pg = pg_to_update

        # get a copy of the fabric data
        #
        fabric = self.fabric.decode(all_details=True,
                                    community_sdr=True,
                                    only_updated=only_updated,
                                    reset_updated=True)

        # the node id of this amfabric
        #
        amfabric_node = ('AMFabric', self.uid)

        # make sure node representing this part of the fabric is in graph
        #
        if amfabric_node not in pg:
            pg.set_node(node=amfabric_node)

            # add edge to represent the setup data
            #
            edge_properties = {'stm_size': self.stm_size,
                               'mp_threshold': self.mp_threshold,
                               'structure': self.structure,
                               'prune_threshold': self.prune_threshold,
                               'ref_id': str_ref_id}

            pg.update_edge(source=amfabric_node, target=amfabric_node, edge=('amfabric_setup', None), **edge_properties)

        # we will store some of the fabric stats in a single self referencing edge between the AMFabric node
        # if this edge is already in the graph then update with changes
        #
        if amfabric_node in pg[amfabric_node] and (('amfabric_stats', None), None) in pg[amfabric_node][amfabric_node]:
            # copy existing properties of edge
            #
            edge_properties = deepcopy(pg[amfabric_node][amfabric_node][(('amfabric_stats', None), None)])

            # update the properties as required
            #
            for attr in fabric:

                # if anomaly or motif attributes then only update entries as required
                #
                if attr in ['anomaly', 'motif']:
                    for idx in fabric[attr]:
                        edge_properties[attr][idx] = fabric[attr][idx]

                # this edge will not include neuro_column data
                #
                elif attr != 'neuro_columns':
                    edge_properties[attr] = fabric[attr]
        else:
            # edge does not exist so create the properties as required
            #
            edge_properties = {attr: fabric[attr] for attr in fabric if attr not in ['neuro_columns']}

        # add in the ref_id
        #
        edge_properties['ref_id'] = ref_id

        pg.update_edge(source=amfabric_node, target=amfabric_node, edge=('amfabric_stats', None), **edge_properties)

        # add edge to represent the short term memory data and associated properties
        #
        edge_properties = {'short_term_memory': [deepcopy(sdr.sdr) for sdr in self.short_term_memory], 'ref_id': ref_id}
        pg.update_edge(source=amfabric_node, target=amfabric_node, edge=('amfabric_short_term_memory', None), **edge_properties)

        # add edge to represent the hebbian edge types seen so far
        #
        if not only_updated or self.non_hebbian_edge_types['updated']:
            edge_properties = {'non_hebbian_edge_types': list(self.non_hebbian_edge_types['edges']), 'ref_id': ref_id}
            self.non_hebbian_edge_types['updated'] = False
            pg.update_edge(source=amfabric_node, target=amfabric_node, edge=('amfabric_non_hebbian_edge_types', None), **edge_properties)

        # update the edges to each neuro_column
        #
        for coord in fabric['neuro_columns']:
            nc_node = ('NeuroColumn', '{}:{}'.format(self.uid, coord))

            edge_properties = {}
            if nc_node in pg[amfabric_node] and (('amfabric_neuro_column', None), None) in pg[amfabric_node][nc_node]:
                edge_properties = deepcopy(pg[amfabric_node][nc_node][(('amfabric_neuro_column', None), None)])

            # upsert the changed properties
            #
            edge_properties.update({attr: fabric['neuro_columns'][coord][attr]
                                    for attr in fabric['neuro_columns'][coord]
                                    if attr not in ['neuro_column']})

            # add in the ref_id
            #
            edge_properties['ref_id'] = str_ref_id

            pg.update_edge(source=amfabric_node, target=nc_node, edge=('amfabric_neuro_column', None), **edge_properties)

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

                pg.update_edge(source=source_node, target=target_node, edge=edge_key,
                               prob=prob, numeric=numeric, numeric_min=numeric_min, numeric_max=numeric_max,
                               generalised_edge_uid=fabric['neuro_columns'][coord]['neuro_column'][edge]['edge_uid'],
                               neuro_column=coord,
                               coord=fabric['neuro_columns'][coord]['coord'],
                               amfabric=self.uid,
                               neuron_id=fabric['neuro_columns'][coord]['neuro_column'][edge]['neuron_id'],
                               ref_id=str_ref_id)

        return pg

    def set_persist_graph(self, pg: AMFGraph) -> None:
        """
        method to initialise the fabric from a graph representation (the persistence graph)

        :param pg: the persistence graph
        :return: None
        """

        # restore basic setup properties
        #
        has_setup = pg[('AMFabric', self.uid)][('AMFabric', self.uid)][(('amfabric_setup', None), None)]
        self.stm_size = has_setup['stm_size']
        self.mp_threshold = has_setup['mp_threshold']
        self.structure = has_setup['structure']
        self.prune_threshold = has_setup['prune_threshold']

        # restore short term memory
        #
        self.short_term_memory = []
        has_stm = pg[('AMFabric', self.uid)][('AMFabric', self.uid)][(('amfabric_short_term_memory', None), None)]
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

        # non hebbian edge types seen so far
        #
        has_non_hebbian_edge_types = pg[('AMFabric', self.uid)][('AMFabric', self.uid)][(('amfabric_non_hebbian_edge_types', None), None)]
        self.non_hebbian_edge_types = {'edges': set(has_non_hebbian_edge_types['non_hebbian_edge_types']), 'updated': False}

        # fabric stats
        #
        has_stats = pg[('AMFabric', self.uid)][('AMFabric', self.uid)][(('amfabric_stats', None), None)]
        fabric = {attr: deepcopy(has_stats[attr]) for attr in has_stats if attr[0] != '_' and attr != 'ref_id'}

        # now neuro_columns
        #
        fabric['neuro_columns'] = {}

        for target in pg[('AMFabric', self.uid)]:
            if target != ('AMFabric', self.uid):
                for edge in pg[('AMFabric', self.uid)][target]:
                    coord_key = '{}:{}'.format(pg[('AMFabric', self.uid)][target][edge]['coord'][0], pg[('AMFabric', self.uid)][target][edge]['coord'][1])

                    # get the attributes for the neuro_column
                    #
                    fabric['neuro_columns'][coord_key] = {attr: pg[('AMFabric', self.uid)][target][edge][attr]
                                                          for attr in pg[('AMFabric', self.uid)][target][edge]
                                                          if attr[0] != '_' and attr not in ['community_nc', 'ref_id']}

                    # correct types from lists to sets or tuples
                    #
                    fabric['neuro_columns'][coord_key]['coord'] = tuple(fabric['neuro_columns'][coord_key]['coord'])
                    fabric['neuro_columns'][coord_key]['nn'] = set(fabric['neuro_columns'][coord_key]['nn'])

                    # place holder for neuro_column
                    #
                    fabric['neuro_columns'][coord_key]['neuro_column'] = {}

                    # fill out community sdr
                    #
                    fabric['neuro_columns'][coord_key]['community_nc'] = SDR()
                    for community_edge_key in pg[('AMFabric', self.uid)][target][edge]['community_nc']:
                        source_node = (pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['source_type'],
                                       pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['source_uid'])
                        target_node = (pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['target_type'],
                                       pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['target_uid'])
                        community_edge = (pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['edge_type'],
                                          pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['edge_uid'])

                        prob = pg[('AMFabric', self.uid)][target][edge]['community_nc'][community_edge_key]['prob']

                        fabric['neuro_columns'][coord_key]['community_nc'].set_item(source_node=source_node,
                                                                                    target_node=target_node,
                                                                                    edge=community_edge,
                                                                                    probability=prob)
        # loop fills out neuro_columns as stacks of sdrs
        #
        for source in pg:

            # ignore edges with AMFabric and NeuroColumn source nodes
            #
            if source[0] not in ['AMFabric', 'NeuroColumn']:
                for target in pg[source]:
                    for edge in pg[source][target]:
                        edge_attr = pg[source][target][edge]
                        coord_key = '{}:{}'.format(edge_attr['coord'][0], edge_attr['coord'][1])

                        # add sdr for this neuron_id if required
                        #
                        if edge_attr['neuron_id'] not in fabric['neuro_columns'][coord_key]['neuro_column']:
                            fabric['neuro_columns'][coord_key]['neuro_column'][edge_attr['neuron_id']] = SDR()

                        fabric['neuro_columns'][coord_key]['neuro_column'][edge_attr['neuron_id']].set_item(source_node=source,
                                                                                                            target_node=target,
                                                                                                            edge=(edge[0][0], edge_attr['generalised_edge_uid']),
                                                                                                            probability=edge_attr['_prob'],
                                                                                                            numeric=edge_attr['_numeric'],
                                                                                                            numeric_min=edge_attr['_numeric_min'],
                                                                                                            numeric_max=edge_attr['_numeric_max'],
                                                                                                            )

        # create NeuralFabric
        self.fabric = NeuralFabric(uid=self.uid,
                                   max_short_term_memory=self.stm_size,
                                   mp_threshold=self.mp_threshold,
                                   structure=self.structure,
                                   prune_threshold=self.prune_threshold)

        # and restore
        #
        self.fabric.restore(fabric)
