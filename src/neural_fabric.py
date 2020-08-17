#!/usr/bin/env python
# -*- encoding: utf-8 -*-

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
    mp_window_size = cython.declare(cython.int, visibility='public')
    mp_window = cython.declare(list, visibility='public')
    anomaly = cython.declare(dict, visibility='public')
    motif = cython.declare(dict, visibility='public')
    anomaly_threshold = cython.declare(cython.double, visibility='public')
    motif_threshold = cython.declare(cython.double, visibility='public')
    mapped = cython.declare(cython.int, visibility='public')
    sum_distance = cython.declare(cython.double, visibility='public')
    mean_distance = cython.declare(cython.double, visibility='public')
    structure = cython.declare(str, visibility='public')

    def __init__(self, uid: str, max_short_term_memory: cython.int = 1, mp_threshold: cython.int = 5, structure: str = 'star'):
        """
        class to represent the columns of neurons in the associative memory fabric.
        Each column is keyed by an x, y coordinate pair key and consists of an sdr
        """

        self.uid = uid
        """ unique name for this area of the fabric """

        self.neurons = {}
        """ the NeuroColumns keyed by coordinates"""

        self.max_stm = max_short_term_memory
        """ the maximum short term memory allowed """

        self.mp_window_size = mp_threshold * self.max_stm
        """ the size of the window used to determine motifs and anomalies """

        self.mp_window = []
        """ window of last mp_window_size distances to the BMU """

        self.anomaly = {}
        """ the anomalies detected so far, keyed by red_id """

        self.motif = {}
        """ the motifs detected so far, keyed by ref_id"""

        self.anomaly_threshold = 0.0
        """ the current distance threshold above which an anomaly is detected """

        self.motif_threshold = 1.0
        """ the current distance threshold below which a motif is detected """

        self.mapped = 0
        """ the number of times data has been mapped to this fabric"""

        self.sum_distance = 0.0
        """ the sum of the BMU distances to mapped data so far """

        self.mean_distance = 0.0
        """ the mean of the BMU distances to mapped data so far """

        self.structure = structure
        """ a string representing the fabric layout structure - 'star' a central neuron with 4 neighbours, 'box' consists of a central neuron with 8 neighbours """

    def seed_fabric(self, example_neuro_column: NeuroColumn, coords: set, hebbian_edges: set):
        """
        method to initialise the fabric with randomised sdrs whose edges and values depend on
        the example NeuroColumn given

        :param example_neuro_column: The example NeuroColumn
        :param coords: set of coordinate tuples to seed
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
            neuro_column = NeuroColumn()
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
                                       'community_nc': NeuroColumn(),
                                       'community_label': None,
                                       'updated': False,
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

    def distance_to_fabric(self, neuro_column: NeuroColumn, ref_id: str = None, bmu_search_filters: set = None) -> tuple:
        """
        method to calculate the distance of sdr to every neuron on the fabric

        :param neuro_column: the NeuroColumn to compare
        :param ref_id: provide a reference id if this search is part of training and matrix profile will be updated
        :param bmu_search_filters: edge types, source node types or target node types to ignore during distance calculation
        :return: a tuple of the neuron distances and path of reasoning structures
        """
        # declare variable types to help cython
        #
        fabric_dist: dict = {}
        fabric_por: dict = {}
        distance: cython.double
        por: list
        bmu_dist: cython.double = 1.0
        bmu_coord_key: Optional[str] = None
        coord_key: str
        anomaly: bool
        motif: bool

        for coord_key in self.neurons:
            distance, por = self.neurons[coord_key]['neuro_column'].calc_distance(neuro_column=neuro_column, filter_types=bmu_search_filters)
            fabric_dist[coord_key] = {'distance': distance,
                                      'last_bmu': self.neurons[coord_key]['last_bmu']}
            fabric_por[coord_key] = por
            if fabric_dist[coord_key]['distance'] <= bmu_dist:
                bmu_dist = fabric_dist[coord_key]['distance']
                bmu_coord_key = coord_key

        # if we have a ref_id then we can update the matrix profile
        #
        anomaly = False
        motif = False
        if ref_id is not None:
            anomaly, motif = self.detect_anomaly_motif(bmu_coord_key=bmu_coord_key, distance=bmu_dist, por=fabric_por[bmu_coord_key], ref_id=ref_id)

        return bmu_coord_key, bmu_dist, anomaly, motif, fabric_dist, fabric_por

    def detect_anomaly_motif(self, bmu_coord_key: str, distance: float, por: list, ref_id: str) -> tuple:
        """
        method to detect if an anomaly or motif has occurred based on the recent max and min bmu distances. This is an implementation of matrix profile

        :param bmu_coord_key: str - the bmu cord key
        :param distance: double - bmu distance
        :param por: - list of distance por records
        :param ref_id: str - the reference id of this update
        :return: tuple of bools with format (anomaly, motif)
        """

        # establish if this is an anomaly or motif
        #
        anomaly = False
        motif = False
        if self.mapped >= 2 * self.max_stm:

            if self.motif_threshold is not None and self.anomaly_threshold is not None:
                # check if this is a new low distance indicating a motif
                #
                if distance <= self.motif_threshold:

                    self.motif[ref_id] = {'bmu_coord': bmu_coord_key, 'distance': distance, 'threshold': self.motif_threshold, 'por': por, 'updated': True}
                    motif = True

                if distance >= self.anomaly_threshold:
                    self.anomaly[ref_id] = {'bmu_coord': bmu_coord_key, 'distance': distance, 'threshold': self.anomaly_threshold, 'por': por, 'updated': True}
                    anomaly = True

            self.mp_window.append(distance)

            # maintain sliding window
            #
            if len(self.mp_window) > self.mp_window_size:
                self.mp_window.pop(0)

            self.motif_threshold = min(self.mp_window)
            self.anomaly_threshold = max(self.mp_window)
        return anomaly, motif

    @cython.ccall
    def update_bmu_stats(self, bmu_coord_key: str, distance: float):
        """
        method to update the stats of the bmu and its neighbours

        :param bmu_coord_key: str - the bmu coordinates
        :param distance: double - the bmu distance
        :return: None
        """

        # declare variable types to help cython
        #
        nn_key: str

        # update the fabric properties
        #
        self.mapped += 1
        self.sum_distance += distance
        self.mean_distance = self.sum_distance / self.mapped

        # update the bmu neuron properties
        #
        self.neurons[bmu_coord_key]['n_bmu'] += 1
        self.neurons[bmu_coord_key]['last_bmu'] = self.mapped
        self.neurons[bmu_coord_key]['sum_distance'] = distance
        self.neurons[bmu_coord_key]['mean_distance'] = self.neurons[bmu_coord_key]['sum_distance'] / self.neurons[bmu_coord_key]['n_bmu']
        self.neurons[bmu_coord_key]['updated'] = True

        for nn_key in self.neurons[bmu_coord_key]['nn']:
            self.neurons[nn_key]['n_nn'] += 1
            self.neurons[nn_key]['last_nn'] = self.mapped
            self.neurons[nn_key]['updated'] = True


    @cython.ccall
    def learn(self, neuro_column: NeuroColumn, bmu_coord: str, coords: list, learn_rates: list, hebbian_edges: set = None):
        """
        hebbian updates the neurons specified using learning rates

        :param neuro_column: the NeuroColumn to learn
        :param bm_coord: the BMU coord
        :param coords: the neuron coordinates to update
        :param learn_rates: the list of learn rates fro each neuron
        :param hebbian_edges: a set of edges to perfrom hebbian learning. If none then all edges will be hebbian learnt
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
        keys_to_process: set
        coord_key: str
        max_weight: cython.double
        target_neuron: str
        bmu_neuron: str = '{}:{}'.format(self.uid, bmu_coord_key)
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
            source_neuron = '{}:{}'.format(self.uid, coord_key)
            bmu_coord_nc.upsert(edge_type='in_community',
                                source_type='neuron', source_uid=source_neuron,
                                target_type='neuron', target_uid=bmu_neuron,
                                neuron_id=0,
                                prob=1.0)

            self.neurons[coord_key]['community_nc'].learn(neuro_column=bmu_coord_nc, learn_rate=learn_rate)

            self.neurons[coord_key]['updated'] = True

            curr_community_edge = self.neurons[coord_key]['community_nc'].get_edge_by_max_probability()
            if curr_community_edge is not None:

                # update the community label for this column of neurons
                #
                self.neurons[coord_key]['community_label'] = curr_community_edge['target_uid']

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

        # the SDR to hold the merged data
        #
        merged_column: NeuroColumn = NeuroColumn()

        idx: cython.int
        coord_key: str
        merge_factor: cython.double

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

    def decode(self, coords: set = None, all_details: bool = True, only_updated: bool = False, community_sdr: bool = False) -> dict:

        if all_details:
            fabric = {'mp_window': self.mp_window,
                      'anomaly': {ref_id: {attr: self.anomaly[ref_id]
                                           for attr in self.anomaly[ref_id]
                                           if attr != 'updated'}
                                  for ref_id in self.anomaly
                                  if not only_updated or self.anomaly['updated']},
                      'motif': {ref_id: {attr: self.motif[ref_id]
                                         for attr in self.motif[ref_id]
                                         if attr != 'updated'}
                                for ref_id in self.motif
                                if only_updated or self.motif['updated']},
                      'anomaly_threshold': self.anomaly_threshold,
                      'motif_threshold': self.motif_threshold,
                      'mapped': self.mapped,
                      'sum_distance': self.sum_distance,
                      'mean_distance': self.mean_distance,
                      'structure': self.structure,
                      'neuro_columns': {}
                      }

            if only_updated:
                for ref_id in fabric['anomaly']:
                    self.anomaly[ref_id]['updated'] = False
                for ref_id in fabric['motif']:
                    self.motif[ref_id]['updated'] = False

        else:
            fabric = {'neuro_columns': {}}

        if coords is None:
            coords_to_decode = self.neurons.keys()
        else:
            coords_to_decode = coords

        for coord_key in coords_to_decode:
            if only_updated:
                self.neurons[coord_key]['updated'] = False

            fabric['neuro_columns'][coord_key] = {n_attr: self.neurons[coord_key][n_attr]
                                                  for n_attr in self.neurons[coord_key]
                                                  if n_attr not in ['neuro_column', 'community_nc', 'updated']}
            fabric['neuro_columns'][coord_key]['neuro_column'] = self.neurons[coord_key]['neuro_column'].decode(only_updated)
            if community_sdr:
                fabric['neuro_columns'][coord_key]['community_nc'] = self.neurons[coord_key]['community_nc'].decode(only_updated)

        return fabric
