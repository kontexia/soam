#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from typing import Optional
from src.neuro_column import NeuroColumn
import cython

@cython.cclass
class NeuralFabric:

    # declare types for instance variables to help cython
    #
    uid = cython.declare(str, visibility='public')
    neurons = cython.declare(dict, visibility='public')
    max_stm = cython.declare(cython.int, visibility='public')
    mp_threshold = cython.declare(cython.double, visibility='public')
    mp_window = cython.declare(list, visibility='public')
    anomaly = cython.declare(dict, visibility='public')
    motif = cython.declare(dict, visibility='public')
    anomaly_threshold = cython.declare(cython.double, visibility='public')
    motif_threshold = cython.declare(cython.double, visibility='public')
    mapped = cython.declare(cython.int, visibility='public')
    sum_distance = cython.declare(cython.double, visibility='public')
    mean_distance = cython.declare(cython.double, visibility='public')
    std_distance = cython.declare(cython.double, visibility='public')
    sum_similarity = cython.declare(cython.double, visibility='public')
    mean_similarity = cython.declare(cython.double, visibility='public')
    std_similarity = cython.declare(cython.double, visibility='public')
    communities = cython.declare(dict, visibility='public')
    structure = cython.declare(str, visibility='public')
    prune_threshold = cython.declare(cython.double, visibility='public')

    def __init__(self,
                 uid: str,
                 max_short_term_memory: cython.int = 1,
                 mp_threshold: float = 0.1,
                 structure: str = 'star',
                 prune_threshold: float = 0.00001):
        """
        class to represent the columns of neurons in the associative memory fabric.
        Each column is keyed by an x, y coordinate pair key and consists of an collection of sdrs

        :param uid: str unique name foe this area of the fabric
        :param max_short_term_memory: the maximum number of neurons in a column of neurons
        :param mp_threshold: the matrix profile noise threshold
        :param structure: str - 'star' structure each column has 4 neighbours  or 'square' structure where each column has 8 neighbours
        :param prune_threshold: float - threshold below which learnt edges are assumed to have zero probability and removed
        """

        self.uid = uid
        """ unique name for this area of the fabric """

        self.neurons = {}
        """ the NeuroColumns keyed by coordinates"""

        self.max_stm = max_short_term_memory
        """ the maximum short term memory allowed """

        self.mp_threshold = mp_threshold
        """ the noise threshold used to determine motifs and anomalies """

        self.mp_window = []
        """ window of last mp_window_size distances to the BMU """

        self.anomaly = {}
        """ the anomalies detected so far, keyed by red_id """

        self.motif = {}
        """ the motifs detected so far, keyed by ref_id"""

        self.anomaly_threshold = -1.0
        """ the current distance threshold above which an anomaly is detected """

        self.motif_threshold = -1.0
        """ the current distance threshold below which a motif is detected """

        self.mapped = 0
        """ the number of times data has been mapped to this fabric"""

        self.sum_distance = 0.0
        """ the sum of the BMU distances to mapped data so far """

        self.mean_distance = 0.0
        """ the mean of the BMU distances to mapped data so far """

        self.std_distance = 0.0
        """ the sample stdev of the BMU distances to mapped data so far """

        self.sum_similarity = 0.0
        """ the sum of the BMU similarities to mapped data so far """

        self.mean_similarity = 0.0
        """ the mean of the BMU similarities to mapped data so far """

        self.std_similarity = 0.0
        """ the sample stdev of the BMU similarity to mapped data so far """

        self.structure = structure
        """ a string representing the fabric layout structure - 'star' a central neuron with 4 neighbours, 'square' consists of a central neuron with 8 neighbours """

        self.prune_threshold = prune_threshold
        """ the threshold below with an edge probability is assumed to be zero and will be deleted """

        self.communities = {}
        """ communities of neurons """

    def seed_fabric(self, example_neuro_column: NeuroColumn, coords: set, hebbian_edges: set):
        """
        method to initialise the fabric with randomised sdrs whose edges and values depend on
        the example NeuroColumn given

        :param example_neuro_column: The example NeuroColumn
        :param coords: set of coordinate tuples to seed
        :param hebbian_edges: set of edge_type that will be hebbian learnt

        :return: None
        """

        # declare variable types to help cython
        #
        x: cython.int
        y: cython.int
        coord_keys_in_fabric: set
        coord: tuple
        coord_key: str
        nn_coord_key: str
        neuro_column: NeuroColumn

        # get a set of the coord keys that will be in the fabric
        #
        coord_keys_in_fabric = {'{}:{}'.format(coord[0], coord[1]) for coord in coords}
        coord_keys_in_fabric.update({coord_key for coord_key in self.neurons})

        for coord in coords:

            # the randomly created sdr
            #
            neuro_column = NeuroColumn(prune_threshold=self.prune_threshold)
            neuro_column.randomize(example_neuro_column, edges_to_randomise=hebbian_edges)

            # key to identify the coordinates of this column of neurons
            #
            coord_key = '{}:{}'.format(coord[0], coord[1])

            # add a new column of neurons
            #
            self.neurons[coord_key] = {'neuro_column': neuro_column,
                                       'coord': coord,
                                       'n_bmu': 0,
                                       'n_nn': 0,
                                       'last_bmu': 0,
                                       'last_nn': 0,
                                       'sum_distance': 0.0,
                                       'mean_distance': 0.0,
                                       'sum_similarity': 0.0,
                                       'mean_similarity': 0.0,
                                       'community_nc': NeuroColumn(prune_threshold=self.prune_threshold),
                                       'community_label': None,
                                       'community_label_prob': 0.0,
                                       'updated': True,     # set ot True so that it will be decoded at least once
                                       'nn': {}
                                       }

            # connect coord to neighbours either with a star or box configuration
            #
            if self.structure == 'star':
                self.neurons[coord_key]['nn'] = {'{}:{}'.format(coord[0] + x, coord[1] + y)
                                                 for x in range(-1, 2)
                                                 for y in range(-1, 2)
                                                 if (((x == 0 and y != 0) or (y == 0 and x != 0)) and
                                                     '{}:{}'.format(coord[0] + x, coord[1] + y) in coord_keys_in_fabric)}
            else:
                self.neurons[coord_key]['nn'] = {'{}:{}'.format(coord[0] + x, coord[1] + y)
                                                 for x in range(-1, 2)
                                                 for y in range(-1, 2)
                                                 if ((coord[0] + x, coord[1] + y) != coord and
                                                     '{}:{}'.format(coord[0] + x, coord[1] + y) in coord_keys_in_fabric)}

        # connect neighbours to coord
        #
        for coord in coords:
            coord_key = '{}:{}'.format(coord[0], coord[1])

            # now connect the neighbours to this new column
            #
            for nn_coord_key in self.neurons[coord_key]['nn']:
                self.neurons[nn_coord_key]['nn'].add(coord_key)

    def grow(self, example_neuro_column: NeuroColumn, coord_key: str = None, hebbian_edges: set = None):
        """
        method to grow the neural fabric

        :param example_neuro_column: the sdr to randomly initialise from
        :param coord_key: str - the coord to grow from
        :param hebbian_edges: set - set of edge_types that are hebbian learnt and needed to randomly initialise new neuro_columns
        :return: None
        """
        # declare variable types to help cython
        #
        coords_to_add: set
        x: cython.int
        y: cython.int
        coord: tuple

        if coord_key is None:
            coord = (0, 0)
        else:

            # get the coord tuple
            # and work out which mini columns to add
            #
            coord = self.neurons[coord_key]['coord']

        # the coords to add depends on the configuration of neighbours
        # star configuration has 4 neighbours
        #
        if self.structure == 'star':
            coords_to_add = {(coord[0] + x, coord[1] + y)
                             for x in range(-1, 2)
                             for y in range(-1, 2)
                             if (((x == 0 and y != 0) or (y == 0 and x != 0)) and
                                 (coord_key is None or '{}:{}'.format(coord[0] + x, coord[1] + y) not in self.neurons[coord_key]['nn']))}

        # else box configuration - eight neighbours
        #
        else:
            coords_to_add = {(coord[0] + x, coord[1] + y)
                             for x in range(-1, 2)
                             for y in range(-1, 2)
                             if ((coord[0] + x, coord[1] + y) != coord and
                                 (coord_key is None or '{}:{}'.format(coord[0] + x, coord[1] + y) not in self.neurons[coord_key]['nn']))}

        if coord_key is None:
            coords_to_add.add(coord)

        # seed the new columns
        #
        self.seed_fabric(example_neuro_column=example_neuro_column, coords=coords_to_add, hebbian_edges=hebbian_edges)

    def distance_to_fabric(self, neuro_column: NeuroColumn, ref_id: str = None, edge_type_filters: set = None, neuron_id_filters: set = None, bmu_only: bool = True) -> dict:
        """
        method to calculate the distance of sdr to every neuron on the fabric

        :param neuro_column: the NeuroColumn to compare
        :param ref_id: provide a reference id if this search is part of training and matrix profile will be updated
        :param edge_type_filters: edge types to compare during distance calculation
        :param neuron_id_filters: neuron_ids to compare during distance calculation
        :param bmu_only: if true the faster search algorithm is used that finds the closest neurocolumn that has been a bmu first and then checks its neighbours if they are closer.
                        If False then a brute force search is used
        :return: a tuple of the neuron distances and path of reasoning structures
        """
        # declare variable types to help cython
        #
        fabric_dist: dict = {}
        distance: cython.double
        similarity: cython.double
        por: dict
        bmu_dist: cython.double = float('inf')
        bmu_similarity: cython.double = 0.0
        bmu_coord_key: Optional[str] = None
        new_bmu_coord_key: Optional[str] = None
        coord_key: str
        anomaly: bool
        motif: bool

        # first search previous bmus
        #
        for coord_key in self.neurons:
            if not bmu_only or (self.mapped == 0 or self.neurons[coord_key]['n_bmu'] > 0):
                distance, similarity, por = self.neurons[coord_key]['neuro_column'].calc_distance(neuro_column=neuro_column,
                                                                                                  edge_type_filters=edge_type_filters,
                                                                                                  neuron_id_filters=neuron_id_filters)
                fabric_dist[coord_key] = {'distance': distance,
                                          'similarity': similarity,
                                          'last_bmu': self.neurons[coord_key]['last_bmu'],
                                          'por': por}
                if fabric_dist[coord_key]['distance'] <= bmu_dist:
                    bmu_dist = fabric_dist[coord_key]['distance']
                    bmu_similarity = fabric_dist[coord_key]['similarity']
                    bmu_coord_key = coord_key

        if bmu_only:
            new_bmu_coord_key = None
            for coord_key in self.neurons[bmu_coord_key]['nn']:
                if coord_key not in fabric_dist:
                    distance, similarity, por = self.neurons[coord_key]['neuro_column'].calc_distance(neuro_column=neuro_column,
                                                                                                      edge_type_filters=edge_type_filters,
                                                                                                      neuron_id_filters=neuron_id_filters)
                    fabric_dist[coord_key] = {'distance': distance,
                                              'similarity': similarity,
                                              'last_bmu': self.neurons[coord_key]['last_bmu'],
                                              'por': por}
                    if fabric_dist[coord_key]['distance'] <= bmu_dist:
                        new_bmu_coord_key = coord_key

            if new_bmu_coord_key is not None:
                bmu_dist = fabric_dist[new_bmu_coord_key]['distance']
                bmu_similarity = fabric_dist[new_bmu_coord_key]['similarity']
                bmu_coord_key = new_bmu_coord_key

        # if we have a ref_id then we can update the matrix profile
        #
        anomaly = False
        motif = False
        if ref_id is not None:
            anomaly, motif = self.detect_anomaly_motif(bmu_coord_key=bmu_coord_key, distance=bmu_dist, por=fabric_dist[bmu_coord_key]['por'], ref_id=ref_id)

        return {'bmu_coord': bmu_coord_key, 'bmu_distance': bmu_dist, 'bmu_similarity': bmu_similarity,
                'anomaly': anomaly, 'motif': motif, 'fabric_distance': fabric_dist}

    def detect_anomaly_motif(self, bmu_coord_key: str, distance: float, por: dict, ref_id: str) -> tuple:
        """
        method to detect if an anomaly or motif has occurred based on the recent max and min bmu distances. This is an implementation of matrix profile

        :param bmu_coord_key: str - the bmu cord key
        :param distance: double - bmu distance
        :param por: - path of reasoning dict
        :param ref_id: str - the reference id of this update
        :return: tuple of bools with format (anomaly, motif)
        """

        # establish if this is an anomaly or motif
        #
        anomaly: bool = False
        motif: bool = False
        low: cython.double
        high: cython.double
        mp_max: cython.double
        mp_min: cython.double
        mp_range: cython.double
        mp_mid: cython.double

        # maintain sliding window of bmu distance (matrix profile)
        #
        self.mp_window.append(distance)
        if self.max_stm == 1:
            window_size = 20
        else:
            window_size = self.max_stm * 2

        if len(self.mp_window) > window_size:
            self.mp_window.pop(0)

        if self.mapped >= window_size:

            if self.motif_threshold > -1.0 and self.anomaly_threshold > -1.0:
                # check if this is a new low distance indicating a motif
                #
                if distance <= self.motif_threshold:

                    self.motif[ref_id] = {'bmu_coord': bmu_coord_key, 'distance': distance, 'threshold': self.motif_threshold, 'por': por, 'updated': True}
                    motif = True

                if distance >= self.anomaly_threshold:
                    self.anomaly[ref_id] = {'bmu_coord': bmu_coord_key, 'distance': distance, 'threshold': self.anomaly_threshold, 'por': por, 'updated': True}
                    anomaly = True

            # calculate the exponential moving average of the matrix profile
            #
            mp_max = max(self.mp_window)
            mp_min = min(self.mp_window)
            mp_range = (mp_max - mp_min)
            mp_mid = (mp_max + mp_min) / 2.0
            mp_range = mp_range * (1 + self.mp_threshold) / 2.0

            # update the anomaly and motif thresholds
            #
            self.anomaly_threshold = mp_mid + mp_range
            self.motif_threshold = mp_mid - mp_range

        return anomaly, motif

    @cython.ccall
    def update_bmu_stats(self, bmu_coord_key: str, fabric_dist: dict):
        """
        method to update the stats of the bmu and its neighbours

        :param bmu_coord_key: str - the bmu coordinates
        :param distance: double - the bmu distance
        :param similarity: double - the bmu similarity
        :return: None
        """

        # declare variable types to help cython
        #
        nn_key: str
        delta: cython.double
        count: int
        n_mapped: int

        # update the fabric properties
        #
        self.mapped += 1

        if self.mapped >= 20:
            if self.mapped == 20:
                self.sum_distance = fabric_dist[bmu_coord_key]['distance']
                self.mean_distance = fabric_dist[bmu_coord_key]['distance']
                self.sum_similarity = fabric_dist[bmu_coord_key]['similarity']
                self.mean_similarity = fabric_dist[bmu_coord_key]['similarity']

            else:
                count = self.mapped - 20
                delta = self.mean_distance + ((fabric_dist[bmu_coord_key]['distance'] - self.mean_distance) / count)

                self.sum_distance += (fabric_dist[bmu_coord_key]['distance'] - self.mean_distance) * (fabric_dist[bmu_coord_key]['distance'] - delta)
                self.mean_distance = delta

                delta = self.mean_similarity + ((fabric_dist[bmu_coord_key]['similarity'] - self.mean_similarity) / count)
                self.sum_similarity += (fabric_dist[bmu_coord_key]['similarity'] - self.mean_similarity) * (fabric_dist[bmu_coord_key]['similarity'] - delta)
                self.mean_similarity = delta

                if count > 1:
                    self.std_distance = math.sqrt(self.sum_distance / (count - 1))
                    self.std_similarity = math.sqrt(self.sum_similarity / (count - 1))

        # update the bmu neuron properties
        #
        self.neurons[bmu_coord_key]['n_bmu'] += 1
        self.neurons[bmu_coord_key]['last_bmu'] = self.mapped
        n_mapped = (self.neurons[bmu_coord_key]['n_bmu'] + self.neurons[bmu_coord_key]['n_nn'])
        self.neurons[bmu_coord_key]['sum_distance'] += fabric_dist[bmu_coord_key]['distance']
        self.neurons[bmu_coord_key]['mean_distance'] = self.neurons[bmu_coord_key]['sum_distance'] / n_mapped
        self.neurons[bmu_coord_key]['sum_similarity'] += fabric_dist[bmu_coord_key]['similarity']
        self.neurons[bmu_coord_key]['mean_similarity'] = self.neurons[bmu_coord_key]['sum_similarity'] / n_mapped
        self.neurons[bmu_coord_key]['updated'] = True

        for nn_key in self.neurons[bmu_coord_key]['nn']:
            self.neurons[nn_key]['n_nn'] += 1
            self.neurons[nn_key]['last_nn'] = self.mapped
            n_mapped = (self.neurons[bmu_coord_key]['n_bmu'] + self.neurons[bmu_coord_key]['n_nn'])

            # its possible that this neighbour isnt in fabric_dist because it was created after the search so default to the bmu
            if nn_key not in fabric_dist:
                key = bmu_coord_key
            else:
                key = nn_key
            self.neurons[nn_key]['sum_distance'] += fabric_dist[key]['distance']
            self.neurons[nn_key]['mean_distance'] = self.neurons[nn_key]['sum_distance'] / n_mapped
            self.neurons[nn_key]['sum_similarity'] += fabric_dist[key]['similarity']
            self.neurons[nn_key]['mean_similarity'] = self.neurons[nn_key]['sum_similarity'] / n_mapped
            self.neurons[nn_key]['updated'] = True


    @cython.ccall
    def learn(self, neuro_column: NeuroColumn, bmu_coord: str, coords: list, learn_rates: list, hebbian_edges: set = None):
        """
        hebbian updates the neurons specified using learning rates

        :param neuro_column: the NeuroColumn to learn
        :param bm_coord: the BMU coord
        :param coords: the neuron coordinates to update
        :param learn_rates: the list of learn rates fro each neuron
        :param hebbian_edges: a set of edges to perform hebbian learning. If none then all edges will be hebbian learnt
        :return: None
        """

        # declare variable types to help cython
        #
        idx: cython.int
        coord_key: str
        learn_rate: cython.double
        bmu: bool

        # assume coords and learning_rates of same length and index position aligns
        #
        for idx in range(len(coords)):
            coord_key = coords[idx]
            if coord_key == bmu_coord:
                bmu = True
            else:
                bmu = False

            learn_rate = learn_rates[idx]
            self.neurons[coord_key]['neuro_column'].learn(neuro_column=neuro_column, learn_rate=learn_rate, hebbian_edges=hebbian_edges, is_bmu=bmu)

    @cython.ccall
    def community_update(self, bmu_coord_key: str, learn_rate: cython.double):
        """
        method to hebbian update the community SDR for the bmu neurons and its neighbours

        :param bmu_coord_key: str - the bmu
        :param learn_rate: double - the learn rate for all updates
        :return: None
        """

        # declare variable types to help cython
        #
        coords_to_update: list
        nn_key: str
        coord_key: str
        bmu_coord_nc: NeuroColumn
        curr_community_edge: dict

        # get list of coordinates to update
        #
        coords_to_update = [nn_key for nn_key in self.neurons[bmu_coord_key]['nn']]
        coords_to_update.append(bmu_coord_key)

        for coord_key in coords_to_update:

            bmu_coord_nc = NeuroColumn()

            # create an sdr to represent the bmu coordinates
            #
            bmu_coord_nc.upsert(edge_type='in_community', edge_uid='',
                                source_type='NeuroColumn', source_uid=coord_key,
                                target_type='NeuroColumn', target_uid=bmu_coord_key,
                                neuron_id=0,
                                prob=1.0)

            self.neurons[coord_key]['community_nc'].learn(neuro_column=bmu_coord_nc, learn_rate=learn_rate)

            self.neurons[coord_key]['updated'] = True

            curr_community_edge = self.neurons[coord_key]['community_nc'].get_edge_by_max_probability()
            if curr_community_edge is not None:

                # remove this coord from the current community if it has changed
                #
                if (self.neurons[coord_key]['community_label'] != curr_community_edge['target_uid'] and
                        self.neurons[coord_key]['community_label'] in self.communities and
                        coord_key in self.communities[self.neurons[coord_key]['community_label']]):

                    self.communities[self.neurons[coord_key]['community_label']].remove(coord_key)
                    if len(self.communities[self.neurons[coord_key]['community_label']]) == 0:
                        del self.communities[self.neurons[coord_key]['community_label']]

                # update the community label for this column of neurons
                #
                self.neurons[coord_key]['community_label'] = curr_community_edge['target_uid']
                self.neurons[coord_key]['community_label_prob'] = curr_community_edge['prob']

                if self.neurons[coord_key]['community_label'] not in self.communities:
                    self.communities[self.neurons[coord_key]['community_label']] = {coord_key}
                else:
                    self.communities[self.neurons[coord_key]['community_label']].add(coord_key)

    def get_fabric_similarity(self, edge_type_filters):

        fabric_sim = {}

        for coord_key in self.neurons:
            if coord_key not in fabric_sim:
                fabric_sim[coord_key] = {'nn': {},
                                         'mean_similarity': 0.0,
                                         'mean_distance': 0.0,
                                         'mean_density': self.neurons[coord_key]['n_bmu'],
                                         'coord': self.neurons[coord_key]['coord'],
                                         'n_bmu': self.neurons[coord_key]['n_bmu'],
                                         'min_similarity': float('inf')
                                         }

            for nn_key in self.neurons[coord_key]['nn']:
                if nn_key in fabric_sim and coord_key in fabric_sim[nn_key]['nn']:
                    fabric_sim[coord_key]['nn'][nn_key] = fabric_sim[nn_key]['nn'][coord_key]
                else:

                    distance, similarity, por = self.neurons[coord_key]['neuro_column'].calc_distance(neuro_column=self.neurons[nn_key]['neuro_column'],
                                                                                                      edge_type_filters=edge_type_filters)
                    fabric_sim[coord_key]['nn'][nn_key] = {'similarity': similarity, 'distance': distance}

                fabric_sim[coord_key]['mean_density'] += self.neurons[nn_key]['n_bmu']
                fabric_sim[coord_key]['mean_similarity'] += fabric_sim[coord_key]['nn'][nn_key]['similarity']
                fabric_sim[coord_key]['mean_distance'] += fabric_sim[coord_key]['nn'][nn_key]['distance']
                if fabric_sim[coord_key]['nn'][nn_key]['similarity'] < fabric_sim[coord_key]['min_similarity']:
                    fabric_sim[coord_key]['min_similarity'] = fabric_sim[coord_key]['nn'][nn_key]['similarity']

            fabric_sim[coord_key]['mean_distance'] = fabric_sim[coord_key]['mean_distance'] / len(self.neurons[coord_key]['nn'])
            fabric_sim[coord_key]['mean_similarity'] = fabric_sim[coord_key]['mean_similarity'] / len(self.neurons[coord_key]['nn'])
            fabric_sim[coord_key]['mean_density'] = fabric_sim[coord_key]['mean_density'] / (len(self.neurons[coord_key]['nn']) + 1)
        return fabric_sim

    def get_fabric_distances(self, edge_type_filters):

        fabric_dist = {}

        for coord_key in self.neurons:
            if coord_key not in fabric_dist:
                fabric_dist[coord_key] = {'nn': {},
                                          'mean_distance': 0.0,
                                          'coord': self.neurons[coord_key]['coord'],
                                          'n_bmu': self.neurons[coord_key]['n_bmu'],
                                          'n_nn': self.neurons[coord_key]['n_nn']}

            for nn_key in self.neurons[coord_key]['nn']:
                if nn_key in fabric_dist and coord_key in fabric_dist[nn_key]['nn']:
                    fabric_dist[coord_key]['nn'][nn_key] = fabric_dist[nn_key]['nn'][coord_key]
                else:

                    distance, similarity, por = self.neurons[coord_key]['neuro_column'].calc_distance(neuro_column=self.neurons[nn_key]['neuro_column'],
                                                                                                      edge_type_filters=edge_type_filters)
                    fabric_dist[coord_key]['nn'][nn_key] = distance
                fabric_dist[coord_key]['mean_distance'] += fabric_dist[coord_key]['nn'][nn_key]

            fabric_dist[coord_key]['n_edges'] = len(self.neurons[coord_key]['nn'])
            fabric_dist[coord_key]['mean_distance'] = fabric_dist[coord_key]['mean_distance'] / fabric_dist[coord_key]['n_edges']

        return fabric_dist

    def merge_neurons(self, coords: list, merge_factors: list) -> NeuroColumn:
        """
        method to create a single SDR from a list of neuron sdrs using a merge_factor

        :param coords: the coordinates of the neurons to merge
        :param merge_factors: the list of merge_factors for each neuron
        :return: merged NeuroColumn
        """
        # declare variable types to help cython
        #
        total_merge_factors: cython.double
        idx: cython.int
        coord_key: str
        merge_factor: cython.double

        # the SDR to hold the merged data
        #
        merged_column: NeuroColumn = NeuroColumn()

        # we will normalise the weights with the total sum of merged factors
        #
        total_merge_factors = sum(merge_factors)
        if total_merge_factors == 0.0:
            total_merge_factors = 1.0

        # assume coords and merge_factors of same length and index position aligns
        #
        for idx in range(len(coords)):
            coord_key = coords[idx]

            # normalise the merge_factor for this neuron
            #
            merge_factor = merge_factors[idx] / total_merge_factors

            merged_column.merge(neuro_column=self.neurons[coord_key]['neuro_column'], merge_factor=merge_factor)
        return merged_column

    def decode(self, coords: set = None, all_details: bool = True, only_updated: bool = False, reset_updated: bool = False, community_sdr: bool = False) -> dict:
        """
        method to decode the entire fabric

        :param coords: set of neuro_column coords to decode. if None then all will be decoded
        :param all_details: If True then all fabric properties will be included else if False just the NeuroColumns
        :param only_updated: If True only the changed data willbe included else if False then all data
        :param reset_updated: If True the update flags will be reset to False else if False the update flags left as is
        :param community_sdr: If True then the community sdr is included
        :return: dictionary representation of the fabric properties
        """

        ref_id: str
        attr: str
        fabric: dict
        coords_to_decode: set
        coord_key: str
        n_attr: str

        if all_details:
            fabric = {'mp_window': self.mp_window,
                      'anomaly': {ref_id: {attr: self.anomaly[ref_id][attr]
                                           for attr in self.anomaly[ref_id]
                                           if attr != 'updated'}
                                  for ref_id in self.anomaly
                                  if not only_updated or self.anomaly[ref_id]['updated']},
                      'motif': {ref_id: {attr: self.motif[ref_id][attr]
                                         for attr in self.motif[ref_id]
                                         if attr != 'updated'}
                                for ref_id in self.motif
                                if only_updated or self.motif[ref_id]['updated']},
                      'anomaly_threshold': self.anomaly_threshold,
                      'motif_threshold': self.motif_threshold,
                      'mapped': self.mapped,
                      'sum_distance': self.sum_distance,
                      'mean_distance': self.mean_distance,
                      'sum_similarity': self.sum_similarity,
                      'mean_similarity': self.mean_similarity,
                      'neuro_columns': {},
                      'communities': {community: {nc for nc in self.communities[community]} for community in self.communities}
                      }

            if reset_updated:
                for ref_id in fabric['anomaly']:
                    self.anomaly[ref_id]['updated'] = False
                for ref_id in fabric['motif']:
                    self.motif[ref_id]['updated'] = False

        else:
            fabric = {'neuro_columns': {}}

        if coords is None:
            coords_to_decode = set(self.neurons.keys())
        else:
            coords_to_decode = coords

        for coord_key in coords_to_decode:
            if not only_updated or self.neurons[coord_key]['updated']:
                if reset_updated:
                    self.neurons[coord_key]['updated'] = False

                # convert sets to lists
                #
                fabric['neuro_columns'][coord_key] = {n_attr: (list(self.neurons[coord_key][n_attr])
                                                               if isinstance(self.neurons[coord_key][n_attr], set)
                                                               else self.neurons[coord_key][n_attr])
                                                      for n_attr in self.neurons[coord_key]
                                                      if n_attr not in ['neuro_column', 'community_nc', 'updated']}

                fabric['neuro_columns'][coord_key]['neuro_column'] = self.neurons[coord_key]['neuro_column'].decode(only_updated)
                if community_sdr:
                    fabric['neuro_columns'][coord_key]['community_nc'] = self.neurons[coord_key]['community_nc'].decode(only_updated)

        return fabric

    def restore(self, fabric: dict) -> None:
        """
        method to restore all the properties of the fabric from a dictionary representation

        :param fabric: the dictionary representation
        :return: None
        """
        self.mp_window = fabric['mp_window']
        self.anomaly = fabric['anomaly']
        self.anomaly_threshold = fabric['anomaly_threshold']

        self.motif = fabric['motif']
        self.motif_threshold = fabric['motif_threshold']
        self.mapped = fabric['mapped']
        self.sum_distance = fabric['sum_distance']
        self.mean_distance = fabric['mean_distance']
        self.sum_similarity = fabric['sum_similarity']
        self.mean_similarity = fabric['mean_similarity']
        self.communities = fabric['communities']

        for coord_key in fabric['neuro_columns']:
            self.neurons[coord_key] = {'neuro_column': NeuroColumn(prune_threshold=self.prune_threshold),
                                       'coord': fabric['neuro_columns'][coord_key]['coord'],
                                       'n_bmu': fabric['neuro_columns'][coord_key]['n_bmu'],
                                       'n_nn': fabric['neuro_columns'][coord_key]['n_nn'],
                                       'last_bmu': fabric['neuro_columns'][coord_key]['last_bmu'],
                                       'last_nn': fabric['neuro_columns'][coord_key]['last_nn'],
                                       'sum_distance': fabric['neuro_columns'][coord_key]['sum_distance'],
                                       'mean_distance': fabric['neuro_columns'][coord_key]['mean_distance'],
                                       'sum_similarity': fabric['neuro_columns'][coord_key]['sum_similarity'],
                                       'mean_similarity': fabric['neuro_columns'][coord_key]['mean_similarity'],
                                       'community_nc': NeuroColumn(prune_threshold=self.prune_threshold),
                                       'community_label': fabric['neuro_columns'][coord_key]['community_label'],
                                       'community_label_prob': fabric['neuro_columns'][coord_key]['community_label_prob'],
                                       'updated': False,
                                       'nn': fabric['neuro_columns'][coord_key]['nn']
                                       }

            # init each neuro_column
            #
            for neuron_id in fabric['neuro_columns'][coord_key]['neuro_column']:
                self.neurons[coord_key]['neuro_column'].upsert_sdr(sdr=fabric['neuro_columns'][coord_key]['neuro_column'][neuron_id], neuron_id=neuron_id)

            # init the community neuro_column
            #
            self.neurons[coord_key]['community_nc'].upsert_sdr(sdr=fabric['neuro_columns'][coord_key]['community_nc'], neuron_id=0)

