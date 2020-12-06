#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional
from copy import deepcopy
from src.subgraph import SubGraph
import numpy as np
from sklearn.cluster import DBSCAN
import logging


def create_fabric(fabric_uid: str,
                  anomaly_threshold: float = 4.0,
                  slow_learn_rate: float = 0.1,
                  fast_learn_rate: float = 0.7,
                  ltm_attention_heads: int = 3,
                  prune_threshold: float = 0.01,
                  generalisation_sensitivity: float = 1.0,
                  normalise_groups: dict = None) -> dict:
    """
    function to create the basic fabric structure
    
    :param fabric_uid: a unique identifier of this part of the fabric
    :param anomaly_threshold: the number of standard deviations in error above which is considered an anomaly
    :param slow_learn_rate: the learning rate to adjust the mapping errors and standard deviations
    :param fast_learn_rate: the learning rate used to adjust each neuron's distance threshold and the fastest learning rate for long-term-memory
    :param ltm_attention_heads: the number of attention heads used to learn sequences
    :param prune_threshold: the probability threshold below which an edge is considered not to exist and thus deleted
    :param generalisation_sensitivity: 1.0 = standard, below 1.0 decreases generalisation and potentially over-fitting, above 1.0 increases generalisation and thus potentially under-fitting 
    :param normalise_groups: a dictionary mapping edge types to a normalisation group - allows different edge types to be normalised together 
    :return: fabric dictionary
    """

    fabric = {
        # the unique identify for this fabric
        #
        'uid': fabric_uid,

        # the domains of neural gas
        #
        'domains': {},

        # a factor for adjusting the generalisation
        #
        'generalise_factor': generalisation_sensitivity,

        # threshold for edge probabilities below which assumed to be zero and deleted - speed optimisation as reduces number of edges to
        # search through and learn
        #
        'prune_threshold': prune_threshold,

        # the alpha used for fast learning
        #
        'fast_learn_rate': fast_learn_rate,

        # the alpha used for slow learning
        #
        'slow_learn_rate': slow_learn_rate,

        # the number of exponential moving averages that maintain the long term memory of a sequence
        #
        'attention_heads': ltm_attention_heads,

        # a SubGraph of the long term memory of all training sub graphs seen so far
        #
        'long_term_memory': None,

        # dictionary of edge_types and corresponding normalisation group
        #
        'normalise_groups': {},

        # unique ID for each update to the fabric
        #
        'update_id': 0,

        # number of standard deviations for anomaly detection
        #
        'anomaly_stdev_threshold': anomaly_threshold,

        # flag indicates attributes at this level of the fabric have changed
        #
        'updated': True
      }

    if normalise_groups is not None:
        fabric['normalise_groups'] = normalise_groups

    logging.debug(f'created fabric: {fabric}')
    
    return fabric


def add_domain(fabric: dict, 
               domain: str) -> None:
    """
    function to add a new domain to a fabric
    
    :param fabric: the fabric to add the domain to
    :param domain: the name of the new domain
    :return: None
    """
    
    fabric['domains'][domain] = {
        # the number of times this domain have been trained
        #
        'mapped': 0,

        # the next neuron ID to be used in this domain
        #
        'next_neuron_id': 0,

        # the neuron id that was the last BMU
        #
        'last_bmu_id': None,

        # the exponential moving average of the mapping error for this domain
        #
        'ema_error': None,

        # the exponential moving average of the variance of the error for this domain
        #
        'ema_variance': 0.0,

        # the current error threshold above which would be considered an anomaly
        #
        'anomaly_threshold': 0.0,

        # dictionary of anomalies keyed by ref id of the update
        #
        'anomalies': {},

        # the error threshold below wich would be considered a motif
        #
        'motif_threshold': float('inf'),

        # dictionary of motifs keyed by ref id of the update
        #
        'motifs': {},

        # dictionary of neurons in domain keyed by neuron id
        #
        'neurons': {},

        # dictionary of mins and maxes keyed by group of edge types
        #
        'sdr_edge_min_max': {},

        # edge types used for finding the bmu
        #
        'search_edge_types': set(),

        # flag to indicate the attributes at this level have changed
        #
        'updated': True
        }

    logging.debug(f'added domain:{domain} to fabric {fabric["uid"]}')


def add_neuron(fabric: dict, 
               domain: str, 
               sub_graph: SubGraph,  
               distance_threshold: float) -> str:
    """
    function adds a new neruon to the fabric domain
    
    :param fabric: the fabric to update
    :param domain: the domain to update
    :param sub_graph: the SubGraph the neuron represents
    :param distance_threshold: the initial distance threshold for this neuron
    :return: the id of the new neuron
    """

    # get the next neuron id
    #
    new_id = str(fabric['domains'][domain]['next_neuron_id'])
    fabric['domains'][domain]['next_neuron_id'] += 1

    neuron = {

        # sub_graph that represents the learned generalisation of edges
        #
        'generalisation': SubGraph(sub_graph),

        # the number of times this neuron has been a BMU
        #
        'n_bmu': 1,

        # the update stamp when this neuron was the last BMU
        #
        'last_bmu': fabric['update_id'],

        # the number of times this neuron has been a runner up
        #
        'n_runner_up': 0,

        # the update stamp when this neuron was last a runner up
        #
        'last_runner_up': None,

        # the exponential moving average mapping error for this neuron
        #
        'ema_error': 0.0,

        # the learned edges to other neurons
        #
        'synapses': SubGraph(),

        # the community label fro this neuron
        #
        'community': 0,

        # the error threshold for this neuron - below which this neuron is considered a BMU
        #
        'threshold': fabric['generalise_factor'] * distance_threshold,

        # flag to indicate the attributes at this level ahev changed
        #
        'updated': True
        }

    fabric['domains'][domain]['neurons'][new_id] = neuron

    logging.debug(f'added neuron: {new_id} to domain{domain} in fabric {fabric["uid"]}')

    return new_id


def get_generalised_distance(fabric: dict, 
                             domain: str, 
                             sub_graph: SubGraph, 
                             search_edge_types: set, 
                             edge_type_weights: Optional[dict] = None) -> list:
    """
    function to calculate the distances of an sub_graph to all neurons within the domain of a fabric
    :param fabric: the fabric that contains the domain
    :param domain: the domain that constains the neurons
    :param sub_graph: the SubGraph to compare to the neurons
    :param search_edge_types: a set of edge types that will be compared
    :param edge_type_weights: a dictionary mapping edge types to a weight - if None then weight assumed to be equal
    :return: a list of tuples containing the neuron_id, the distance, the path of reasoning explaining the contribution of each edge
    """
    distances = []

    for neuron_id in fabric['domains'][domain]['neurons']:
        distance, por = fabric['domains'][domain]['neurons'][neuron_id]['generalisation'].calc_euclidean_distance(sub_graph=sub_graph,
                                                                                                                  compare_edge_types=search_edge_types,
                                                                                                                  edge_type_weights=edge_type_weights)
        distances.append((neuron_id, distance, por, fabric['domains'][domain]['neurons'][neuron_id]['n_bmu']))

    # sort in neurons ascending order of distance and descending order of n_bmu (in case two or more have same distance prefer the most updated)
    #
    distances.sort(key=lambda x: (x[1], -x[3]))

    logging.debug(f'calculated distance against {len(fabric["domains"][domain]["neurons"])} neurons in domain: {domain} fabric {fabric["uid"]}')

    return distances


def update_domain_error(fabric: dict, 
                        domain: str,
                        bmu_id: str,
                        error: float,
                        ref_id: str,
                        new_id: Optional[str] = None,
                        ) -> tuple:
    """
    function to update the domain's mapping error and to detect anomalies or motifs 
    :param fabric: the fabric to update
    :param domain: the domain to update
    :param bmu_id: the id of the BMU neuron
    :param new_id: the id of the new neuron
    :param error: the distance error of the incoming SubGraph to the BMU
    :param ref_id: the external reference id to log with an anomaly or motif
    :return: tuple containing booleans: (IsAnomaly, IsMotif)
    """

    # update the exponential moving average error - note first value equals first error recording
    #
    if fabric['domains'][domain]['ema_error'] is None:
        fabric['domains'][domain]['ema_error'] = error
    else:
        fabric['domains'][domain]['ema_error'] += (error - fabric['domains'][domain]['ema_error']) * fabric['slow_learn_rate']

    fabric['domains'][domain]['ema_variance'] += (pow(error - fabric['domains'][domain]['ema_error'], 2) - fabric['domains'][domain]['ema_variance']) * fabric['slow_learn_rate']

    # record breaches of anomaly threshold
    #
    report: dict = {'bmu_id': bmu_id, 'new_id': new_id, 'mapped': fabric['domains'][domain]['mapped'], 'error': error}
    anomaly = False
    motif = False

    if error > fabric['domains'][domain]['anomaly_threshold']:
        fabric['domains'][domain]['anomalies'][ref_id] = report
        anomaly = True
    elif error <= fabric['domains'][domain]['motif_threshold']:
        fabric['domains'][domain]['motifs'][ref_id] = report
        motif = True

    # update thresholds for next training data
    #
    stdev = np.sqrt(fabric['domains'][domain]['ema_variance'])
    fabric['domains'][domain]['anomaly_threshold'] = fabric['domains'][domain]['ema_error'] + (fabric['anomaly_stdev_threshold'] * stdev)
    fabric['domains'][domain]['motif_threshold'] = max(fabric['domains'][domain]['ema_error'] - (2 * stdev), 0.0)

    fabric['domains'][domain]['updated'] = True

    logging.debug(f'updated mapping errors for domain: {domain} fabric: {fabric["uid"]} anomaly: {anomaly} motif: {motif}')

    return anomaly, motif


def update_synapse(fabric: dict, 
                   learn_rate: float,
                   source_domain: str, 
                   source_id: str, 
                   target_domain: str, 
                   target_id: str, 
                   edge_type: str,
                   edge_uid: Optional[str] = None
                   ) -> None:
    """
    function to update the synapse sub_graph of a neuron with an edge
    :param fabric: the fabric that contains the domain
    :param learn_rate: the learn rate to learn the edge
    :param source_domain: the domain that contains the source neuron
    :param source_id: the id of the source neuron
    :param target_domain: the domain that contains the target neuron 
    :param target_id: the id of the target neuron
    :param edge_type: the type of edge to create
    :param edge_uid: the uid of the edge to create
    :return: None
    """

    # create a sub graph to learn
    #
    sub_graph = SubGraph()
    sub_graph.set_item(source_node=(source_domain, source_id),
                       edge=(edge_type, edge_uid),
                       target_node=(target_domain, target_id))

    fabric['domains'][source_domain]['neurons'][source_id]['synapses'].learn(sub_graph=sub_graph,
                                                                             learn_rate=learn_rate,
                                                                             learn_edge_types={edge_type},
                                                                             prune_threshold=fabric['prune_threshold'])

    fabric['domains'][target_domain]['neurons'][target_id]['updated'] = True

    logging.debug(f'updated edge: ({edge_type},{edge_uid}) source neuron: ({source_domain},{source_id}) target neuron: ({target_domain},{target_id})')


def update_generalisation(fabric: dict, 
                          domain: str, 
                          neuron_id: str, 
                          sub_graph: SubGraph,
                          learn_rate: float, 
                          learn_edge_types: set) -> None:
    """
    function updates the generalisation for a neuron
    :param fabric: the fabric that contains the domain
    :param domain: the domain that contains the neuron
    :param neuron_id: the neuron id
    :param sub_graph: the sub_graph to learn
    :param learn_rate: 
    :param learn_edge_types: 
    :return: 
    """

    fabric['domains'][domain]['neurons'][neuron_id]['generalisation'].learn(sub_graph=sub_graph,
                                                                            learn_rate=learn_rate,
                                                                            learn_edge_types=learn_edge_types,
                                                                            prune_threshold=fabric['prune_threshold'])

    fabric['domains'][domain]['neurons'][neuron_id]['updated'] = True

    logging.debug(f'updated generalised graph for neuron:{neuron_id} in domain: {domain} in fabric {fabric["uid"]}')


def normalise_sub_graph(fabric: dict,
                        domain: str,
                        sub_graph: SubGraph) -> SubGraph:
    """
    function normalises any edge in the sub graph that has a numeric value - the fabric keeps track of the max and min values seen so far and will re-normalise
    any neuron generalisation if the max or mins are exceeded  
    :param fabric: the fabric that contains the domain
    :param domain: the domain that contains the neurons that this sub graph will be normailsed for
    :param sub_graph: the sub graph to normalise
    :return: a normalised sub graph
    """

    norm_sg = SubGraph(sub_graph=sub_graph)

    renormalise_edges = set()

    for edge in norm_sg:
        if norm_sg[edge]['numeric'] is not None:

            # if this edge type isnt in the normalise groups dict then add
            #
            if norm_sg[edge]['edge_type'] not in fabric['normalise_groups']:
                fabric['normalise_groups'][norm_sg[edge]['edge_type']] = norm_sg[edge]['edge_type']

            group = fabric['normalise_groups'][norm_sg[edge]['edge_type']]

            # update the global min and max for this edge
            #
            if group not in fabric['domains'][domain]['sdr_edge_min_max']:
                fabric['domains'][domain]['sdr_edge_min_max'][group] = {'min': norm_sg[edge]['numeric'] - 0.001, 'max': norm_sg[edge]['numeric'],
                                                                        'prev_min': norm_sg[edge]['numeric'] - 0.001, 'prev_max': norm_sg[edge]['numeric']}

            elif norm_sg[edge]['numeric'] < fabric['domains'][domain]['sdr_edge_min_max'][group]['min']:
                fabric['domains'][domain]['sdr_edge_min_max'][group]['prev_min'] = fabric['domains'][domain]['sdr_edge_min_max'][group]['min']
                fabric['domains'][domain]['sdr_edge_min_max'][group]['prev_max'] = fabric['domains'][domain]['sdr_edge_min_max'][group]['max']

                fabric['domains'][domain]['sdr_edge_min_max'][group]['min'] = norm_sg[edge]['numeric']
                renormalise_edges.add(edge)

            elif norm_sg[edge]['numeric'] > fabric['domains'][domain]['sdr_edge_min_max'][group]['max']:
                fabric['domains'][domain]['sdr_edge_min_max'][group]['prev_min'] = fabric['domains'][domain]['sdr_edge_min_max'][group]['min']
                fabric['domains'][domain]['sdr_edge_min_max'][group]['prev_max'] = fabric['domains'][domain]['sdr_edge_min_max'][group]['max']
                fabric['domains'][domain]['sdr_edge_min_max'][group]['max'] = norm_sg[edge]['numeric']
                renormalise_edges.add(edge)

            # normalise the numeric
            #
            norm_sg[edge]['numeric'] = ((norm_sg[edge]['numeric'] - fabric['domains'][domain]['sdr_edge_min_max'][group]['min']) /
                                         (fabric['domains'][domain]['sdr_edge_min_max'][group]['max'] - fabric['domains'][domain]['sdr_edge_min_max'][group]['min']))

    # if the global mins and maxes have changed then need to renormalise existing neurons
    #
    if len(renormalise_edges) > 0:
        # the domain min and max must have changed
        #
        fabric['domains'][domain]['updated'] = True

        for neuron in fabric['domains'][domain]['neurons']:

            # get out the neuron's generalised sub_graph
            #
            n_sg = fabric['domains'][domain]['neurons'][neuron]['generalisation']

            # for each edge that has changed
            #
            for edge in renormalise_edges:
                if edge in n_sg and n_sg[edge]['numeric'] is not None:

                    # this neuron is changing
                    #
                    fabric['domains'][domain]['neurons'][neuron]['updated'] = True

                    group = fabric['normalise_groups'][n_sg[edge]['edge_type']]

                    # first denormalise using previous min and max
                    #
                    denorm_numeric = ((n_sg[edge]['numeric'] *
                                       (fabric['domains'][domain]['sdr_edge_min_max'][group]['prev_max'] - fabric['domains'][domain]['sdr_edge_min_max'][group]['prev_min'])) +
                                      fabric['domains'][domain]['sdr_edge_min_max'][group]['prev_min'])

                    # then normalise using new min and max
                    #
                    n_sg[edge]['numeric'] = ((denorm_numeric - fabric['domains'][domain]['sdr_edge_min_max'][group]['min']) /
                                              (fabric['domains'][domain]['sdr_edge_min_max'][group]['max'] - fabric['domains'][domain]['sdr_edge_min_max'][group]['min']))

    return norm_sg


def denormalise_sub_graph(fabric: dict,
                          domain: str,
                          sub_graph: SubGraph) -> SubGraph:
    """
    function to denormalise a sub graph based on the max and mins of numeric edges seen in the domain
    :param fabric: the fabric that contains the domain
    :param domain: the domain that contains the max and mins
    :param sub_graph: the sub graph to denormalise
    :return: the denormalised sub graph
    """

    # create a new copy
    #
    denorm_sg = SubGraph(sub_graph=sub_graph)

    for edge in denorm_sg:

        # if the edge has been normalised then it will be present in fabric['domains'][domain]['sdr_edge_min_max']
        #
        if denorm_sg[edge]['numeric'] is not None and denorm_sg[edge]['edge_type'] in fabric['normalise_groups']:
            group = fabric['normalise_groups'][denorm_sg[edge]['edge_type']]
            denorm_sg[edge]['numeric'] = ((denorm_sg[edge]['numeric'] *
                                           (fabric['domains'][domain]['sdr_edge_min_max'][group]['max'] - fabric['domains'][domain]['sdr_edge_min_max'][group]['min'])) +
                                          fabric['domains'][domain]['sdr_edge_min_max'][group]['min'])
    return denorm_sg


def decode_fabric(fabric: dict) -> dict:
    """
    function to copy the fabric, calculate the communities for each domain of neurons and denormalise all neurons
    :param fabric: the fabric to copy
    :return: the decoded fabric
    """

    decoded_fabric = deepcopy(fabric)
    for domain in decoded_fabric['domains']:

        # update the community for the domain before denormalising sdrs
        #
        update_community(fabric=decoded_fabric,
                         domain=domain,
                         search_edge_types=decoded_fabric['domains'][domain]['search_edge_types'])

        # denormalise the generalised sdrs
        #
        for neuron_id in decoded_fabric['domains'][domain]['neurons']:
            decoded_fabric['domains'][domain]['neurons'][neuron_id]['generalisation'] = denormalise_sub_graph(fabric=fabric,
                                                                                                              domain=domain,
                                                                                                              sub_graph=decoded_fabric['domains'][domain]['neurons'][neuron_id]['generalisation'])

    return decoded_fabric


def update_community(fabric: dict,
                     domain: str,
                     search_edge_types: set) -> None:
    """
    function to calculate the community of neurons within a domain
    :param fabric: the fabric that contains the domain
    :param domain: the domain that contains the neurons
    :param search_edge_types: set of edge types to use in the distance calculation between neurons
    :return: None
    """

    if len(fabric['domains'][domain]['neurons']) > 3:
        neuron_idx = 0
        neuron_id_map = {}
        for neuron_id in fabric['domains'][domain]['neurons']:
            neuron_id_map[neuron_id] = neuron_idx
            neuron_idx += 1

        # matrix of distances between all neurons within the domain
        #
        distances = [[-1.0] * len(fabric['domains'][domain]['neurons'])
                     for _ in range(len(fabric['domains'][domain]['neurons']))]
        sum_distance = 0.0
        n_distances = 0.0
        for neuron_id_1 in fabric['domains'][domain]['neurons']:
            for neuron_id_2 in fabric['domains'][domain]['neurons']:
                if neuron_id_1 != neuron_id_2:
                    if distances[neuron_id_map[neuron_id_2]][neuron_id_map[neuron_id_1]] == -1.0:
                        neuron_1_sdr = fabric['domains'][domain]['neurons'][neuron_id_1]['generalisation']
                        neuron_2_sdr = fabric['domains'][domain]['neurons'][neuron_id_2]['generalisation']

                        nn_distance, _ = neuron_1_sdr.calc_euclidean_distance(sub_graph=neuron_2_sdr, compare_edge_types=search_edge_types)

                        distances[neuron_id_map[neuron_id_1]][neuron_id_map[neuron_id_2]] = nn_distance
                        distances[neuron_id_map[neuron_id_2]][neuron_id_map[neuron_id_1]] = nn_distance
                        sum_distance += nn_distance
                        n_distances += 1
                else:
                    distances[neuron_id_map[neuron_id_1]][neuron_id_map[neuron_id_2]] = 0.0
                    distances[neuron_id_map[neuron_id_2]][neuron_id_map[neuron_id_1]] = 0.0

        dist_matrix = np.array(distances)
        communities = DBSCAN(eps=sum_distance / n_distances, min_samples=3).fit(dist_matrix)
        for neuron_id in fabric['domains'][domain]['neurons']:
            fabric['domains'][domain]['neurons'][neuron_id]['community'] = communities.labels_[neuron_id_map[neuron_id]]


def train_domain(fabric: dict,
                 domain: str,
                 sub_graph: SubGraph,
                 ref_id: str,
                 search_edge_types: set,
                 learn_edge_types: set,
                 edge_type_weights: Optional[dict] = None,
                 learn_nearest_neighbours: bool = True) -> dict:
    """
    method trains a fabric domain with a sub graph

    :param fabric: the fabric that contains the domain
    :param domain: the domain to train
    :param sub_graph: the sub graph to train the domain
    :param ref_id: the reference id for this action
    :param search_edge_types: the set of edge types to used to find the BMU neuron
    :param learn_edge_types: the set of edge types that will be learnt by the BMU neuron
    :param edge_type_weights: an optional dictionary mapping edge types to weights used in the search for the BMU neuron
    :param learn_nearest_neighbours: boolean flag if True will force the learning of the nearest neighbour to the BMU neuron
    :return: path of reasoning dictionary
    """

    # add the domain of not existing
    #
    if domain not in fabric['domains']:
        add_domain(fabric=fabric, domain=domain)

    # create a new unique update_id
    #
    fabric['update_id'] += 1
    fabric['updated'] = True

    logging.debug(f'training domain: {domain} in fabric {fabric["uid"]} for ref id: {ref_id} and update id: {fabric["update_id"]}')

    # keep track of the number of times this domain has been mapped to
    #
    fabric['domains'][domain]['mapped'] += 1
    fabric['domains'][domain]['updated'] = True

    # keep track of the edge types used to find the bmu
    #
    fabric['domains'][domain]['search_edge_types'].update(search_edge_types)

    # prepare a path of reasoning
    #
    por = {
        # the external reference for this update
        #
        'ref_id': ref_id,
        domain: {
            # the number of times this domain has been mapped
            #
            'mapped': fabric['domains'][domain]['mapped'],

            # the neuron id of the BMU
            #
            'bmu_id': None,

            # the learning rate used to modify the BMU
            #
            'bmu_learn_rate': None,

            # the error threshold of the BMU
            #
            'bmu_threshold': None,

            # the ID of a neruon that has just been created
            #
            'created_id': None,

            # the neuron id of the runner up neuron
            #
            'runner_up_id': None,

            # the learning rate used to modify the runner up
            #
            'runner_up_learn_rate': None,

            # the path of reasoning resulting from searching for the BMU
            #
            'search_por': None,

            # the distance to the BMU
            #
            'bmu_distance': None,

            # the exponential moving average of the domain error
            #
            'ema_error': None,

            # set the anomaly and motif thresholds to values prior to update
            #
            'anomaly_threshold': fabric['domains'][domain]['anomaly_threshold'],
            'motif_threshold': fabric['domains'][domain]['motif_threshold']

        }
           }

    # if the fabric in the required domain is empty then just add a new neuron
    #
    if len(fabric['domains'][domain]['neurons']) == 0:

        # add new neuron and set threshold to 0.0 to make it easy for next neuron to be added
        #
        new_id = add_neuron(fabric=fabric, domain=domain, sub_graph=sub_graph, distance_threshold=0.0)

        # remember this neuron id was last bmu
        #
        fabric['domains'][domain]['last_bmu_id'] = new_id

        # por needs to know this was a new neuron
        #
        por[domain]['created_id'] = new_id

    else:

        # calc the distances of incoming data to all existing neurons in the required domain
        #
        distances = get_generalised_distance(fabric=fabric,
                                             domain=domain,
                                             sub_graph=sub_graph,
                                             search_edge_types=search_edge_types,
                                             edge_type_weights=edge_type_weights)

        # the bmu is the closest neuron
        #
        bmu_id = distances[0][0]
        por[domain]['bmu_id'] = bmu_id

        logging.debug(f'found BMU neuron:{bmu_id} in domain: {domain} of fabric: {fabric["uid"]}')

        bmu_distance = distances[0][1]

        por[domain]['bmu_distance'] = bmu_distance
        por[domain]['search_por'] = distances[0][2]
        por[domain]['bmu_threshold'] = fabric['domains'][domain]['neurons'][bmu_id]['threshold']

        # if the distance exceeds the bmu threshold then add a new neuron
        #
        if bmu_distance > fabric['domains'][domain]['neurons'][bmu_id]['threshold']:

            new_id = add_neuron(fabric=fabric, domain=domain, sub_graph=sub_graph, distance_threshold=bmu_distance)
            por[domain]['created_id'] = new_id

            # adjust the existing bmu threshold to make it harder to insert a new neuron next to it
            #
            fabric['domains'][domain]['neurons'][bmu_id]['threshold'] += (((bmu_distance * fabric['generalise_factor']) - fabric['domains'][domain]['neurons'][bmu_id]['threshold']) *
                                                                          fabric['fast_learn_rate'])

            # remember the last bmu for this domain
            #
            fabric['domains'][domain]['last_bmu_id'] = new_id

        else:

            # no new neuron is created so set to None
            #
            new_id = None

            # update the number of times the bmu neuron has been mapped
            #
            fabric['domains'][domain]['neurons'][bmu_id]['n_bmu'] += 1

            # keep track of the last time it was mapped to
            #
            fabric['domains'][domain]['neurons'][bmu_id]['last_bmu'] = fabric['update_id']

            # keep track of the average error of mapped data
            #
            fabric['domains'][domain]['neurons'][bmu_id]['ema_error'] += (bmu_distance - fabric['domains'][domain]['neurons'][bmu_id]['ema_error']) * fabric['slow_learn_rate']

            # calculate the learning rate for this neuron
            #
            bmu_learn_rate = (1 / fabric['domains'][domain]['neurons'][bmu_id]['n_bmu'])
            por[domain]['bmu_learn_rate'] = bmu_learn_rate

            # adjust the threshold of the existing bmu
            #
            fabric['domains'][domain]['neurons'][bmu_id]['threshold'] += (((bmu_distance * fabric['generalise_factor']) - fabric['domains'][domain]['neurons'][bmu_id]['threshold']) *
                                                                          fabric['fast_learn_rate'])

            # learn the generalisations
            #
            update_generalisation(fabric=fabric,
                                  domain=domain,
                                  neuron_id=bmu_id,
                                  sub_graph=sub_graph,
                                  learn_rate=bmu_learn_rate,
                                  learn_edge_types=learn_edge_types)

            # process the runner up if required
            #
            if learn_nearest_neighbours and len(distances) > 1:

                runner_up_id = distances[1][0]
                runner_up_distance = distances[1][1]
                if runner_up_distance <= fabric['domains'][domain]['neurons'][runner_up_id]['threshold']:

                    por[domain]['runner_up_id'] = runner_up_id

                    logging.debug(f'found Runner up neuron: {runner_up_id} in domain {domain} in fabric {fabric["uid"]}')

                    runner_up_learn_rate = min(bmu_learn_rate,  (1 / fabric['domains'][domain]['neurons'][runner_up_id]['n_bmu']))
                    por[domain]['runner_up_learn_rate'] = runner_up_learn_rate

                    fabric['domains'][domain]['neurons'][runner_up_id]['threshold'] += (((runner_up_distance * fabric['generalise_factor']) -
                                                                                         fabric['domains'][domain]['neurons'][runner_up_id]['threshold']) *
                                                                                        fabric['fast_learn_rate'])

                    # map sub_graph to the runner up
                    #
                    update_generalisation(fabric=fabric,
                                          domain=domain,
                                          neuron_id=runner_up_id,
                                          sub_graph=sub_graph,
                                          learn_rate=runner_up_learn_rate,
                                          learn_edge_types=learn_edge_types)

                    # remember last time it was a runner up
                    #
                    fabric['domains'][domain]['neurons'][runner_up_id]['last_runner_up'] = fabric['update_id']

                    # update the number of times the runner_up neuron has been runner up
                    #
                    fabric['domains'][domain]['neurons'][runner_up_id]['n_runner_up'] += 1

                    # update nearest neighbour edges
                    #
                    update_synapse(fabric=fabric,
                                   source_domain=domain, source_id=bmu_id,
                                   target_domain=domain, target_id=runner_up_id,
                                   edge_type='nn',
                                   learn_rate=bmu_learn_rate)

            # remember the last bmu for this domain
            #
            fabric['domains'][domain]['last_bmu_id'] = bmu_id

        por[domain]['anomaly'], por[domain]['motif'] = update_domain_error(fabric=fabric,
                                                                           domain=domain,
                                                                           bmu_id=bmu_id,
                                                                           new_id=new_id,
                                                                           error=bmu_distance,
                                                                           ref_id=ref_id)
        por[domain]['error'] = bmu_distance
        por[domain]['ema_error'] = fabric['domains'][domain]['ema_error']

    return por


def learn_spatial(fabric: dict,
                  ref_id: str,
                  sub_graph: SubGraph,
                  search_edge_types: set,
                  learn_edge_types: set) -> dict:
    """
    function to train the SPATIAL domain of a fabric
    :param fabric: the fabric to train
    :param ref_id: the reference id for this action
    :param sub_graph: the sub graph to train the SPATIAL domain
    :param search_edge_types: the set of edge types to use to find the BMU neuron
    :param learn_edge_types: the set of edge types to learn by the BMU neuron
    :return: path of reasoning dictionary
    """

    # add the SPATIAL domain if it doesn't exist
    #
    if 'SPATIAL' not in fabric['domains']:
        add_domain(fabric=fabric, domain='SPATIAL')

    # first train the main spatial domain
    #
    spatial_norm_sdr = normalise_sub_graph(fabric=fabric, domain='SPATIAL', sub_graph=sub_graph)
    por = train_domain(fabric=fabric,
                       domain='SPATIAL',
                       sub_graph=spatial_norm_sdr,
                       ref_id=ref_id,
                       search_edge_types=search_edge_types,
                       learn_edge_types=learn_edge_types,
                       learn_nearest_neighbours=True)

    return por


def learn_temporal(fabric: dict,
                   ref_id: str,
                   sub_graph: SubGraph,
                   search_edge_types: set,
                   learn_edge_types: set) -> dict:
    """
    function to train the TEMPORAL domain of a fabric with a sequence of sub graphs. It does this by maintaining an infinite filter memory of sub_graphs seen
    stored in a single sub graph with n attention heads of edges that are learned at increasingly slower rates

    :param fabric: the fabric to train
    :param ref_id: the reference id for this action
    :param sub_graph: the sub raph to train the TEMPORAL domain
    :param search_edge_types: the set of edge types to use to find the BMU neuron
    :param learn_edge_types: the set of edge types to learn by the BMU neuron
    :return: path of reasoning dictionary
    """

    # the path of reasoning
    #
    por = {}

    if fabric['long_term_memory'] is not None:
        # the temporal domain will search and learn edges for each attention head
        #
        ltm_search_edge_types = {f'{edge_type}_HEAD_{head}' for edge_type in search_edge_types for head in range(fabric['attention_heads'])}
        ltm_learn_edge_types = {f'{edge_type}_HEAD_{head}' for edge_type in learn_edge_types for head in range(fabric['attention_heads'])}

        # add the TEMPORAL domain if it doesn't exist
        #
        if 'TEMPORAL' not in fabric['domains']:
            add_domain(fabric=fabric,
                       domain='TEMPORAL')

        # normalise the current long_term_memory
        #
        ltm_norm_sg = normalise_sub_graph(fabric=fabric, domain='TEMPORAL', sub_graph=fabric['long_term_memory'])

        # train the temporal domain with the current normalised long_term_memory
        #
        por = train_domain(fabric=fabric,
                           domain='TEMPORAL',
                           sub_graph=ltm_norm_sg,
                           ref_id=ref_id,
                           search_edge_types=ltm_search_edge_types,
                           learn_edge_types=ltm_learn_edge_types,
                           learn_nearest_neighbours=True)

        # learn the edge between the temporal_neuron and the spatial neuron
        #
        spatial_neuron_id = fabric['domains']['SPATIAL']['last_bmu_id']
        temporal_neuron_id = fabric['domains']['TEMPORAL']['last_bmu_id']
        temporal_learn_rate = (1 / fabric['domains']['TEMPORAL']['neurons'][temporal_neuron_id]['n_bmu'])
        update_synapse(fabric=fabric,
                       source_domain='TEMPORAL', source_id=temporal_neuron_id,
                       target_domain='SPATIAL', target_id=spatial_neuron_id,
                       edge_type='temporal_nn',
                       learn_rate=temporal_learn_rate)

    else:
        fabric['long_term_memory'] = SubGraph()

    # update the long_term_memory sub_graph
    # for the number of required attention heads, create a head specific set of edges and train the current temporal sub_graph with a head specific learn_rate
    #
    for head in range(fabric['attention_heads']):
        ltm_sg = SubGraph()

        edge_types_to_learn = set()
        # long_term_memory memorises the denormalised incoming SubGraph
        #
        for edge in sub_graph:

            # temporal sub_graph is only interested in the edge types used to find the spatial domain bmu
            #
            if sub_graph[edge]['edge_type'] in search_edge_types:

                # each head has a specific edge_type
                #
                edge_type = f"{sub_graph[edge]['edge_type']}_HEAD_{head}"
                edge_types_to_learn.add(edge_type)

                # make sure all HEAD edge types are in the same normalisation group
                #
                if sub_graph[edge]['edge_type'] in fabric['normalise_groups']:
                    fabric['normalise_groups'][edge_type] = f"{fabric['normalise_groups'][sub_graph[edge]['edge_type']]}_HEAD"
                else:
                    fabric['normalise_groups'][edge_type] = f"{sub_graph[edge]['edge_type']}_HEAD"

                # copy incoming data, making each edge specific to head
                #
                ltm_sg.set_item(source_node=(sub_graph[edge]['source_type'], sub_graph[edge]['source_uid']),
                                edge=(edge_type, sub_graph[edge]['edge_uid']),
                                target_node=(sub_graph[edge]['target_type'], sub_graph[edge]['target_uid']),
                                probability=sub_graph[edge]['prob'],
                                numeric=sub_graph[edge]['numeric'])

        fabric['long_term_memory'].learn(sub_graph=ltm_sg,

                                         # the learn rate depends on the attention head being trained - it decreases by the power
                                         #
                                         learn_rate=pow(fabric['fast_learn_rate'], head + 1),
                                         learn_edge_types=edge_types_to_learn,
                                         prune_threshold=fabric['prune_threshold'])
    return por


def learn_association(fabric: dict,
                      ref_id: str,
                      domain: str,
                      sub_graph: SubGraph,
                      search_edge_types: set,
                      learn_edge_types: set,
                      spatial_neuron_id: Optional[str] = None,
                      learn_nearest_neighbours: bool = True) -> dict:
    """
    a function to train an association domain that is linked to the SPATIAL domain BMU neuron
    :param fabric: the fabric that contains the association domain
    :param ref_id: the reference id for this action
    :param domain: the association domain to train
    :param sub_graph: the sub graph to train the association domain
    :param search_edge_types: the set of edge types to use to find the BMU neuron
    :param learn_edge_types: the set of edge types to learn by the BMU neuron
    :param spatial_neuron_id: optional SPATIAL domain bmu neuron to associate with the Association domain - if None then previous SPATIAL BMU used
    :param learn_nearest_neighbours: boolean flag to indicate if the nearest neighbour of the association BMU neuron is also learnt
    :return: path of reasoning dictionary
    """

    # add the domain if it doesn't exist
    #
    if domain not in fabric['domains']:
        add_domain(fabric=fabric, domain=domain)

    # will need the spatial neuron id an learning rate
    #
    if spatial_neuron_id is None:
        spatial_neuron_id = fabric['domains']['SPATIAL']['last_bmu_id']
    spatial_learn_rate = (1 / fabric['domains']['SPATIAL']['neurons'][spatial_neuron_id]['n_bmu'])

    # normalise the sub_graph
    #
    norm_sg = normalise_sub_graph(fabric=fabric, domain=domain, sub_graph=sub_graph)
    por = train_domain(fabric=fabric,
                       domain=domain,
                       sub_graph=norm_sg,
                       ref_id=ref_id,
                       search_edge_types=search_edge_types,
                       learn_edge_types=learn_edge_types,
                       learn_nearest_neighbours=learn_nearest_neighbours)

    # learn the connection between the spatial neuron and the association neuron
    #
    neuron_id = fabric['domains'][domain]['last_bmu_id']
    update_synapse(fabric=fabric,
                   source_domain='SPATIAL', source_id=spatial_neuron_id,
                   target_domain=domain, target_id=neuron_id,
                   edge_type='association',
                   learn_rate=spatial_learn_rate)
    return por


def get_association(fabric: dict,
                    domain: str,
                    neuron_id: str,
                    depth: int) -> dict:
    """
    recursive function that will retrieve the association neurons for the given neuron_id within a given domain
    :param fabric: the fabric that contains the domain
    :param domain: the domain that contains the neuron
    :param neuron_id: the neuron id to retrieve the associations for
    :param depth: the depth of associations to find
    :return: dictionary of all associations
    """

    query_result = {'associations': {}}
    total_probabilities = {}

    synapse_sdr = fabric['domains'][domain]['neurons'][neuron_id]['synapses']
    for edge in synapse_sdr:
        if synapse_sdr[edge]['edge_type'] != 'nn':
            association_domain = synapse_sdr[edge]['target_type']

            if association_domain not in query_result['associations']:
                query_result['associations'][association_domain] = []
                total_probabilities[association_domain] = 0.0

            total_probabilities[association_domain] += synapse_sdr[edge]['prob']
            association_neuron = fabric['domains'][association_domain]['neurons'][synapse_sdr[edge]['target_uid']]
            association = {'domain': association_domain,
                           'neuron_id': synapse_sdr[edge]['target_uid'],
                           'probability': synapse_sdr[edge]['prob'],
                           'n_bmu': association_neuron['n_bmu'],
                           'ema_error': association_neuron['ema_error'],
                           'community': association_neuron['community'],
                           'generalisation': denormalise_sub_graph(fabric=fabric,
                                                                   domain=association_domain,
                                                                   sub_graph=association_neuron['generalisation'])
                           }
            if depth - 1 > 0:
                next_associations = get_association(fabric=fabric, domain=association_domain, neuron_id=synapse_sdr[edge]['target_uid'], depth=(depth - 1))
                association.update(next_associations)
            query_result['associations'][association_domain].append(association)

    for association_domain in total_probabilities:
        for association in query_result['associations'][association_domain]:
            association['probability'] /= total_probabilities[association_domain]
        query_result['associations'][association_domain].sort(key=lambda x: x['probability'], reverse=True)

    return query_result


def query_domain(fabric: dict,
                 domain: str,
                 sub_graph: SubGraph,
                 association_depth: int = 1) -> dict:
    """
    function that queries the the specified domain and returns the neuron most similar to the exemplar sub graph
    :param fabric: the fabric that contains the domain
    :param domain: the domain to query
    :param sub_graph: the exemplar sub graph
    :param association_depth: the depth of association to return
    :return: dictionary of the most similar neuron and its' associations
    """

    query_result = {}
    if domain in fabric['domains']:

        # normalise
        #
        norm_sg = normalise_sub_graph(fabric=fabric, domain=domain, sub_graph=sub_graph)
        search_edge_types = {norm_sg[edge]['edge_type'] for edge in norm_sg}
        distances = get_generalised_distance(fabric=fabric, domain=domain, sub_graph=norm_sg, search_edge_types=search_edge_types)

        # the bmu is the closest neuron
        #
        query_result['bmu_id'] = distances[0][0]
        query_result['domain'] = domain
        query_result['distance'] = distances[0][1]
        query_result['por'] = distances[0][2]

        # fill out important attributes about neuron
        #
        query_result['n_bmu'] = fabric['domains'][domain]['neurons'][query_result['bmu_id']]['n_bmu']
        query_result['ema_error'] = fabric['domains'][domain]['neurons'][query_result['bmu_id']]['ema_error']
        query_result['community'] = fabric['domains'][domain]['neurons'][query_result['bmu_id']]['community']

        # create the copy of the generalised graph and denormalise
        #
        query_result['generalisation'] = denormalise_sub_graph(fabric=fabric,
                                                               domain=domain,
                                                               sub_graph=fabric['domains'][domain]['neurons'][query_result['bmu_id']]['generalisation'])

        # provide associations
        #
        if association_depth > 0:
            next_associations = get_association(fabric=fabric, domain=domain, neuron_id=query_result['bmu_id'], depth=association_depth)
            query_result.update(next_associations)
        else:
            query_result['associations'] = {}

    return query_result


def next_in_sequence(fabric: dict) -> dict:
    """
    function to return the predicted sub graph that is next in the sequence given the current long term memory of the fabric
    :param fabric: the fabric to query
    :return: the predicted neuron and its' associations
    """

    # using the current long term memory get the next in sequence
    #
    query_result = query_domain(fabric=fabric, domain='TEMPORAL', sub_graph=fabric['long_term_memory'], association_depth=2)

    return query_result
