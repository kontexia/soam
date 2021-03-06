#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy, copy
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
                 mp_threshold: float = 0.1,
                 structure: str = 'star',
                 prune_threshold: float = 0.00001,
                 min_cluster_size: int = 2,
                 cluster_start_threshold: float = 0.5,
                 cluster_step: float = 0.01,
                 random_seed=None,
                 save_por_history=False) -> None:
        """
        class that implements the associative memory fabric

        :param uid: str unique name foe this area of the fabric
        :param short_term_memory: the maximum number of neurons in a column of neurons
        :param mp_threshold: the noise threshold used to determine anomalies and motifs using matrix profile
        :param structure: str - 'star' structure each column has 4 neighbours  or 'box' structure where each column has 8 neighbours
        :param prune_threshold: float - threshold below which learnt edges are assumed to have zero probability and removed
        :param min_cluster_size: minimum number of neuro_columns allowed in a cluster
        :param cluster_start_threshold: starting similarity threshold
        :param cluster_step: increment in similarity threshold during search
        :param random_seed: a seed to ensure repeatable experiments
        :param save_por_history: If true tge por history will be saved
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
        """ the matrix profile noise threshold """

        self.structure = structure
        """ the structure of the neuro_columns - star - 5 columns connected in a star, square - 9 neuro_columns connected in a square"""

        self.prune_threshold = prune_threshold
        """ threshold below which learnt edges are assumed to have zero probability and removed """

        self.fabric = NeuralFabric(uid=uid, max_short_term_memory=self.stm_size, mp_threshold=mp_threshold, structure=structure, prune_threshold=prune_threshold)
        """ the fabric of neuro-columns """

        self.non_hebbian_edge_types = {'edges': set(), 'updated': True}
        """ the edge_types that are not to be learned using a hebbian rule """

        self.hebbian_edge_types = {'edges': set(), 'updated': True}
        """ the edge types that can be learned using a hebbian rule """

        self.persist_graph = None
        """ the persist graph describing the fabric """

        self.min_cluster_size = min_cluster_size
        """ the minimum number of neuro_columns allowed in a cluster """

        self.cluster_start_threshold = cluster_start_threshold
        """ the similarity theshold to start the search for clusters of neuro_columns """

        self.cluster_step = cluster_step
        """ the increment in similarity threshold used in searching for clusters of neuro_columns """

        self.save_por_history = save_por_history
        """ flag to indicate if the por history must be saved """

        self.por_history = []
        """ the history of por """

        # seed the random number generator if required
        #
        if random_seed is not None:
            random.seed(random_seed)

    def search_for_bmu(self, sdr: SDR, ref_id: Union[str, int], non_hebbian_edges=('generalise',), fast_search: bool = True) -> dict:
        """
        method finds the bmu neuro_column that must be trained and detects if an anomaly or motif has occurred

        :param sdr: sparse data representation of a graph to learn
        :param ref_id: a reference id
        :param non_hebbian_edges: a tuple of edge types identifying edges that will not be learnt using a hebbian rule and not used in the
                                search for the Best Matching Unit
        :param fast_search: If true a fast approximation for finding the BMU is used else if false a brute force search is used
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
        por = {# unique reference for this por
               'uid': '{}:{}'.format(self.uid, ref_id),

               # the fabric
               #
               'fabric': self.uid,

               # the reference id for this update
               #
               'ref_id': str_ref_id,

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

        # update the hebbian edges
        #
        hebbian_edges = {neuro_column[edge_key]['edge_type']
                         for edge_key in neuro_column
                         if neuro_column[edge_key]['edge_type'] not in self.non_hebbian_edge_types['edges']}

        if len(self.hebbian_edge_types['edges'] & hebbian_edges) != len(self.hebbian_edge_types['edges'] | hebbian_edges):
            # keep track of the hebbian edges
            #
            self.hebbian_edge_types['edges'].update(hebbian_edges)
            self.hebbian_edge_types['updated'] = True

        # get set of neuron_ids to search for
        #
        search_neuron_ids = {neuro_column[edge_key]['neuron_id'] for edge_key in neuro_column}

        # if the fabric is empty initialise
        #
        initial_setup = False
        if len(self.fabric.neurons) == 0:
            initial_setup = True
            self.fabric.grow(example_neuro_column=neuro_column, hebbian_edges=self.hebbian_edge_types['edges'])

        # find the BMU by calculating the distance of the NeuroColumn to every column in the fabric
        #
        search_results = self.fabric.distance_to_fabric(neuro_column=neuro_column,
                                                        ref_id=str_ref_id,
                                                        edge_type_filters=self.hebbian_edge_types['edges'],
                                                        neuron_id_filters=search_neuron_ids,
                                                        bmu_only=fast_search)

        # create the por
        #
        if initial_setup:
            # if we have just created the first neurocolumns then force the first bmu to have coordinate 0, 0
            #
            por['bmu'] = '0:0'
            por['bmu_distance'] = 0.0       # by forcing distance to 0.0 /similarity to 1.0 then neuro_column 0:0 will effectively copy the sdr
            por['bmu_similarity'] = 1.0
            por['anomaly'] = search_results['anomaly']
            por['motif'] = search_results['motif']
            por['fabric_distance'] = search_results['fabric_distance']
            por['fabric_distance']['0:0']['distance'] = 0.0
            por['fabric_distance']['0:0']['similarity'] = 1.0

        else:
            # select actual bmu
            #
            por['bmu'] = search_results['bmu_coord']
            por['bmu_distance'] = search_results['bmu_distance']
            por['bmu_similarity'] = search_results['bmu_similarity']
            por['anomaly'] = search_results['anomaly']
            por['motif'] = search_results['motif']
            por['fabric_distance'] = search_results['fabric_distance']

        return por

    def learn(self, search_por: dict, similarity_learn: bool = True) -> dict:
        """
        method that learns the current short term memory taking into account the bmu suggested in search_por
        :param search_por: the por resulting from a previous call to search_for_bmu
        :param similarity_learn: if true then learning rate is proportional to the similarity,
                            else if false learning rate is proportional to distance (1- similarity)
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

        # bmu similarity
        #
        bmu_similarity = search_por['bmu_similarity']

        if similarity_learn:

            # the learn_rate is proportional to the similarity
            #
            bmu_learn_rate = bmu_similarity
        else:
            # the learn_rate is proportional to the distance
            #
            bmu_learn_rate = (1 - bmu_similarity)

        # the distance to all columns
        #
        fabric_dist = search_por['fabric_distance']

        # grow the mini column neighbours if required - note any edge columns will always require new neighbours
        # if selected as the bmu
        #
        existing_coords = set(self.fabric.neurons.keys())
        new_coords = set()
        if ((self.fabric.structure == 'star' and len(self.fabric.neurons[bmu_coord_key]['nn']) < 4) or
            (self.fabric.structure == 'square' and len(self.fabric.neurons[bmu_coord_key]['nn']) < 8)):
            self.fabric.grow(example_neuro_column=neuro_column, coord_key=bmu_coord_key, hebbian_edges=hebbian_edges)
            new_coords = set(self.fabric.neurons.keys()) - existing_coords

        # update the bmu and its neighbours stats
        #
        self.fabric.update_bmu_stats(bmu_coord_key=bmu_coord_key, fabric_dist=fabric_dist)

        # get neighbourhood of neurons to learn
        #
        coords_to_update = list(self.fabric.neurons[bmu_coord_key]['nn'])

        # add in the bmu coord
        #
        coords_to_update.append(bmu_coord_key)

        # and the learn rates for each mini column of neurons. if mini-column has just been grown then won't be in fabric_dist so default to bmu_learn_rate
        #
        if similarity_learn:
            learn_rates = [fabric_dist[coord_key]['similarity'] if coord_key in fabric_dist else bmu_learn_rate
                           for coord_key in coords_to_update]
        else:
            learn_rates = [(1 - fabric_dist[coord_key]['similarity']) if coord_key in fabric_dist else bmu_learn_rate
                           for coord_key in coords_to_update]

        self.fabric.learn(neuro_column=neuro_column, bmu_coord=bmu_coord_key, coords=coords_to_update, learn_rates=learn_rates, hebbian_edges=hebbian_edges)

        learn_por = {'updated_bmu': bmu_coord_key,
                     'updated_bmu_learn_rate': bmu_learn_rate,
                     'updated_nn': [nn for nn in coords_to_update if nn != bmu_coord_key],
                     'updated_nn_learn_rate': [learn_rates[idx]
                                               for idx in range(len(coords_to_update))
                                               if coords_to_update[idx] != bmu_coord_key],
                     'coords_grown': list(new_coords),
                     'mean_distance':  self.fabric.mean_distance,
                     'std_distance': self.fabric.std_distance,
                     }

        # update the history of por
        #
        por = deepcopy(search_por)
        por.update(learn_por)
        self.por_history.append(por)

        return learn_por

    def train(self, sdr: SDR, ref_id: Union[str, int], non_hebbian_edges=('generalise',), fast_search: bool = True, similarity_learn: bool = True) -> dict:
        """
        method trains the neural network with one sdr

        :param sdr: sparse data representation of a graph to learn
        :param ref_id: a reference id
        :param non_hebbian_edges: a tuple of edge types identifying edges that will not be learnt using a hebbian rule and not used in the
                                search for the Best Matching Unit
        :param fast_search: If true a fast approximation for finding the BMU is used else if false a brute force search is used
        :param similarity_learn: if true then learning rate is proportional to the similarity,
                            else if false learning rate is proportional to distance (1- similarity)
        :return: dict - the Path Of Reasoning
        """
        # search for the bmu and calculate anomalies/ motifs
        #
        search_por = self.search_for_bmu(sdr=sdr, ref_id=ref_id, non_hebbian_edges=non_hebbian_edges, fast_search=fast_search)

        # learn the current short term memory given the search_por results
        #
        learn_por = self.learn(search_por=search_por, similarity_learn=similarity_learn)

        # combine pors
        #
        search_por.update(learn_por)
        return search_por

    def query(self, sdr, top_n: int = 1, fast_search: bool = False) -> Dict[str, Union[List[str], NeuroColumn]]:
        """
        method to query the associative memory neural network
        :param sdr: Is a sdr or list of SDRs that will be used to query the network. If a list it is assumed the order provided sequence context
                    where list[0] is oldest and list[n] is most recent. Also assumes list is not bigger than preconfigure sequence_dimension
        :param top_n: int - if > 1 then the top n neuro-columns will be merged weighted by the closeness to the query SDR(s)
        :param fast_search: if true a fast approximation algorithm is used to find the BMU else if False then Brute Force is used

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
        search_neuron_ids = {query_nc[edge_key]['neuron_id'] for edge_key in query_nc}

        # search the fabric
        #
        search_results = self.fabric.distance_to_fabric(neuro_column=query_nc, edge_type_filters=search_edges, neuron_id_filters=search_neuron_ids, bmu_only=fast_search)

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

            # create a list of distances and neuron coords, sort in ascending order of distance and then pick out the first top_n
            #
            distances = [(coord_key, search_results['fabric_distance'][coord_key]['distance'], search_results['fabric_distance'][coord_key]['similarity'])
                         for coord_key in search_results['fabric_distance']]

            distances.sort(key=lambda x: x[1])
            merge_factors = []
            coords_to_merge = []
            n = 0
            idx = 0
            while n < top_n and idx < len(distances):

                coords_to_merge.append(distances[idx][0])
                merge_factors.append(distances[idx][2])
                n += 1
                idx += 1

            # merge the top_n neurons with a merge factor equal to similarity
            #
            merged_neuron = self.fabric.merge_neurons(coords=coords_to_merge, merge_factors=merge_factors)
            result['coords'] = coords_to_merge
            result['neuro_column'] = merged_neuron.decode()

        return result

    def decode_fabric(self, all_details: bool = False, min_cluster_size: Optional[int] = None, start_threshold: Optional[float] = None, step: Optional[float] = None) -> dict:
        """
        method extracts the neural fabric into a dict structure whilst decoding each SDR
        :param all_details: if True will include all details of the fabric
        :param min_cluster_size: minimum number of neuro_columns allowed in a cluster
        :param start_threshold: starting similarity threshold
        :param step: increment in similarity threshold during search
        :return: dictionary keyed by neuron column coordinates containing all the attributes
        """

        if min_cluster_size is None:
            min_cluster_size = self.min_cluster_size
        if start_threshold is None:
            start_threshold = self.cluster_start_threshold
        if step is None:
            step = self.cluster_step

        # make sure communities have been updated
        #
        self.fabric.update_communities(min_cluster_size=min_cluster_size,
                                       start_threshold=start_threshold,
                                       step=step,
                                       edge_type_filters=self.hebbian_edge_types['edges'])

        fabric = self.fabric.decode(all_details=all_details,
                                    only_updated=False,
                                    reset_updated=False)
        return fabric

    def get_communities(self, min_cluster_size: int = None, start_threshold: float = None, step: float = None):
        """
        method to find the minimum number of communities
        :param min_cluster_size: minimum number of neuro_columns allowed in a cluster
        :param start_threshold: starting similarity threshold
        :param step: increment in similarity threshold during search
        :return: None
        """

        if min_cluster_size is None:
            min_cluster_size = self.min_cluster_size
        if start_threshold is None:
            start_threshold = self.cluster_start_threshold
        if step is None:
            step = self.cluster_step

        self.fabric.update_communities(min_cluster_size=min_cluster_size,
                                       start_threshold=start_threshold,
                                       step=step,
                                       edge_type_filters=self.hebbian_edge_types['edges'])

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

    def update_persist_graph(self, ref_id: Union[str, int], pg_to_update: Optional[AMFGraph] = None, only_updated: bool = True):
        """
        method to update a graph representation (persistence graph) of the fabric

        :param ref_id: a unique reference that will be tagged on every changed edge
        :param pg_to_update: If provided this AMFGraph will be updated
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
            if self.persist_graph is None:
                self.persist_graph = AMFGraph()

        else:
            self.persist_graph = pg_to_update

        # get a copy of the fabric data
        #
        fabric = self.fabric.decode(all_details=True,
                                    only_updated=only_updated,
                                    reset_updated=True)

        # the node id of this amfabric
        #
        amfabric_node = ('AMFabric', self.uid)

        # make sure node representing this part of the fabric is in graph
        #
        if amfabric_node not in self.persist_graph:
            self.persist_graph.set_node(node=amfabric_node)

            # add edge to represent the setup data
            #
            edge_properties = {'stm_size': self.stm_size,
                               'mp_threshold': self.mp_threshold,
                               'structure': self.structure,
                               'prune_threshold': self.prune_threshold,
                               'min_cluster_size': self.min_cluster_size,
                               'cluster_start_threshold': self.cluster_start_threshold,
                               'cluster_step': self.cluster_step,
                               'save_por_history': self.save_por_history,
                               'ref_id': str_ref_id}

            self.persist_graph.update_edge(source=amfabric_node, target=('AMFabricSetup', '*'), edge=('has_amfabric_setup', None), prob=1.0, **edge_properties)

        # we will store some of the fabric stats in a single self referencing edge between the AMFabric node
        # if this edge is already in the graph then update with changes
        #
        if amfabric_node in self.persist_graph[amfabric_node] and ('AMFabricStats', '*') in self.persist_graph[amfabric_node]:

            # copy existing properties of edge
            #
            edge_properties = deepcopy(self.persist_graph[amfabric_node][('AMFabricStats', '*')][(('has_amfabric_stats', None), None)])

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

        self.persist_graph.update_edge(source=amfabric_node, target=('AMFabricStats', '*'), edge=('has_amfabric_stats', None), prob=1.0, **edge_properties)

        # add edge to represent the short term memory data and associated properties
        #
        edge_properties = {'short_term_memory': [deepcopy(sdr.sdr) for sdr in self.short_term_memory], 'ref_id': ref_id}
        self.persist_graph.update_edge(source=amfabric_node, target=('AMFabricSTM', '*'), edge=('has_amfabric_stm', None), prob=1.0, **edge_properties)

        # add edge to represent the edge types seen so far
        #
        if not only_updated or self.non_hebbian_edge_types['updated'] or self.hebbian_edge_types['updated']:
            edge_properties = {'non_hebbian': list(self.non_hebbian_edge_types['edges']),
                               'hebbian': list(self.hebbian_edge_types['edges']),
                               'ref_id': ref_id
                               }
            self.non_hebbian_edge_types['updated'] = False
            self.hebbian_edge_types['updated'] = False

            self.persist_graph.update_edge(source=amfabric_node, target=('AMFabricEdgeTypes', '*'), edge=('has_amfabric_edge_types', None), prob=1.0, **edge_properties)

        # update the edges to each neuro_column
        #
        for coord in fabric['neuro_columns']:
            nc_node = ('NeuroColumn', '{}:{}'.format(self.uid, coord))

            edge_properties = {}
            if nc_node in self.persist_graph[amfabric_node] and (('has_amfabric_neuro_column', None), None) in self.persist_graph[amfabric_node][nc_node]:
                edge_properties = deepcopy(self.persist_graph[amfabric_node][nc_node][(('has_amfabric_neuro_column', None), None)])

            # upsert the changed properties
            #
            edge_properties.update({attr: fabric['neuro_columns'][coord][attr]
                                    for attr in fabric['neuro_columns'][coord]
                                    if attr not in ['neuro_column']})

            # add in the ref_id
            #
            edge_properties['ref_id'] = str_ref_id

            self.persist_graph.update_edge(source=amfabric_node, target=nc_node, edge=('has_amfabric_neuro_column', None), prob=1.0, **edge_properties)

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

                self.persist_graph.update_edge(source=source_node, target=target_node, edge=edge_key,
                                               prob=prob, numeric=numeric, numeric_min=numeric_min, numeric_max=numeric_max,
                                               generalised_edge_uid=fabric['neuro_columns'][coord]['neuro_column'][edge]['edge_uid'],
                                               neuro_column=coord,
                                               coord=fabric['neuro_columns'][coord]['coord'],
                                               amfabric=self.uid,
                                               neuron_id=fabric['neuro_columns'][coord]['neuro_column'][edge]['neuron_id'],
                                               ref_id=str_ref_id)

        return self.persist_graph

    def get_por_history(self) -> list:
        """
        method to return history of por and then empties the list
        :return: the por history
        """
        history = copy(self.por_history)
        self.por_history = []
        return history

    def get_persist_graph(self, ref_id: str) -> AMFGraph:
        """
        method to return the current persist_graph
        :return: the persist_graph
        """

        # make sure communities have been updated
        #
        self.fabric.update_communities(min_cluster_size=self.min_cluster_size,
                                       start_threshold=self.cluster_start_threshold,
                                       step=self.cluster_step,
                                       edge_type_filters=self.hebbian_edge_types['edges'])

        self.update_persist_graph(ref_id=ref_id, only_updated=True)

        return self.persist_graph

    def set_persist_graph(self, pg: AMFGraph) -> None:
        """
        method to initialise the fabric from a graph representation (the persistence graph)

        :param pg: the persistence graph
        :return: None
        """

        # make sure the persist graph is rebuilt next time get_persist_graph is called
        #
        self.persist_graph = None

        # restore basic setup properties
        #
        has_setup = pg[('AMFabric', self.uid)][('AMFabricSetup', '*')][(('has_amfabric_setup', None), None)]
        self.stm_size = has_setup['stm_size']
        self.mp_threshold = has_setup['mp_threshold']
        self.structure = has_setup['structure']
        self.prune_threshold = has_setup['prune_threshold']
        self.min_cluster_size = has_setup['min_cluster_size']
        self.cluster_start_threshold = has_setup['cluster_start_threshold']
        self.cluster_step = has_setup['cluster_step']
        self.save_por_history = has_setup['save_por_history']
        self.por_history = []

        # restore short term memory
        #
        self.short_term_memory = []
        has_stm = pg[('AMFabric', self.uid)][('AMFabricSTM', '*')][(('has_amfabric_stm', None), None)]
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

        #  edge types seen so far
        #
        has_edge_types = pg[('AMFabric', self.uid)][('AMFabricEdgeTypes', '*')][(('has_amfabric_edge_types', None), None)]
        self.non_hebbian_edge_types = {'edges': set(has_edge_types['non_hebbian']), 'updated': False}
        self.hebbian_edge_types = {'edges': set(has_edge_types['hebbian']), 'updated': False}

        # fabric stats
        #
        has_stats = pg[('AMFabric', self.uid)][('AMFabricStats', '*')][(('has_amfabric_stats', None), None)]
        fabric = {attr: deepcopy(has_stats[attr]) for attr in has_stats if attr[0] != '_' and attr != 'ref_id'}

        # now neuro_columns
        #
        fabric['neuro_columns'] = {}

        for target in pg[('AMFabric', self.uid)]:
            if target[0] == 'NeuroColumn':
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

        # loop fills out neuro_columns as stacks of sdrs
        #
        for source in pg:

            # ignore edges with AMFabric and NeuroColumn source nodes
            #
            if source[0] not in ['AMFabric', 'NeuroColumn', 'AMFabricStats', 'AMFabricSetup', 'AMFabricSTM', 'AMFabricNonHebbTypes']:
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
