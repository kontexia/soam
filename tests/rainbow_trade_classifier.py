#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
import random

import plotly.graph_objects as go

from src.amfabric import AMFabric
from src.sdr import SDR

import json


def print_anomalies(amf, por_results):
    sp_mp = []
    sp_anomaly = []
    sp_motif = []
    sp_anomaly_threshold = []
    sp_motif_threshold = []

    sp_anomalies = amf.get_anomalies()
    sp_motifs = amf.get_motifs()

    for por in por_results:
        sp_mp.append(por['bmu_distance'])

        sp_anomaly_threshold.append(por['anomaly_threshold'])
        sp_motif_threshold.append(por['motif_threshold'])

        if por['ref_id'] in sp_anomalies:
            sp_anomaly.append(por['bmu_distance'])
        else:
            sp_anomaly.append(None)
        if por['ref_id'] in sp_motifs:
            sp_motif.append(por['bmu_distance'])
        else:
            sp_motif.append(None)

    sp_mp_scatter = go.Scatter(name='Matrix Profile', x=[idx for idx in range(len(sp_mp))], y=sp_mp, mode='lines', line=dict(width=2.0, color='green'))
    sp_anomaly_scatter = go.Scatter(name='anomalies', x=[idx for idx in range(len(sp_anomaly))], y=sp_anomaly, mode='markers+text', marker=dict(size=10, color='red', opacity=0.7))
    sp_motif_scatter = go.Scatter(name='motifs', x=[idx for idx in range(len(sp_motif))], y=sp_motif, mode='markers+text', marker=dict(size=10, color='green', opacity=0.7))

    sp_motif_threshold_scatter = go.Scatter(name='motif threshold', x=[idx for idx in range(len(sp_motif_threshold))], y=sp_motif_threshold, mode='lines', line=dict(width=1.0, color='red'))
    sp_anomaly_threshold_scatter = go.Scatter(name='anomaly threshold', x=[idx for idx in range(len(sp_anomaly_threshold))], y=sp_anomaly_threshold, mode='lines', line=dict(width=1.0, color='red'))

    fig = go.Figure(data=[sp_mp_scatter, sp_anomaly_scatter, sp_motif_scatter, sp_motif_threshold_scatter, sp_anomaly_threshold_scatter])
    fig.update_layout(scene=dict(xaxis_title='REF ID', yaxis_title='Similarity'), width=1400, height=900,
                      title=dict(text='AMFabric Pooler Anomalies & Motifs'))
    fig.show()


def plot_fabric_3d(fabric, title, coords_to_highlight=None, neurons_to_plot=None, short_term_memory=1):
    x_node = []
    y_node = []
    z_node = []
    colors = []
    x_edge = []
    y_edge = []
    z_edge = []
    labels = []
    sizes = []
    pairs = set()
    nos_winners = 0
    sum_distance = 0.0
    sum_mapped = 0

    #
    if neurons_to_plot is None:
        neurons_to_plot = [n for n in range(short_term_memory)]

    if coords_to_highlight is None:
        coords_to_highlight = list(fabric['neuro_columns'].keys())

    for coord_id in fabric['neuro_columns'].keys():

        if fabric['neuro_columns'][coord_id]['n_bmu'] > 0:
            nos_winners += 1
            sum_distance += fabric['neuro_columns'][coord_id]['mean_distance']
            sum_mapped += fabric['neuro_columns'][coord_id]['n_bmu']

        for neuron_id in neurons_to_plot:
            if 'trade:*:{}:{}:{}:{}:{}'.format('has_rgb', 'r', neuron_id, 'rgb', 'r') in fabric['neuro_columns'][coord_id]['neuro_column']:

                # if neuron in the top of the column and has been trained then increase size
                #
                if neuron_id == neurons_to_plot[0] and fabric['neuro_columns'][coord_id]['n_bmu'] + fabric['neuro_columns'][coord_id]['n_nn'] > 0:
                    sizes.append(20 + fabric['neuro_columns'][coord_id]['n_bmu'])
                else:
                    sizes.append(15)

                nc = fabric['neuro_columns'][coord_id]['neuro_column']

                x = fabric['neuro_columns'][coord_id]['coord'][0]
                y = fabric['neuro_columns'][coord_id]['coord'][1]
                z = max(neurons_to_plot) - neuron_id

                x_node.append(x)
                y_node.append(y)
                z_node.append(z)

                if coord_id in coords_to_highlight:
                    r = round(nc['trade:*:{}:{}:{}:{}:{}'.format('has_rgb', 'r', neuron_id, 'rgb', 'r')]['numeric'])
                    g = round(nc['trade:*:{}:{}:{}:{}:{}'.format('has_rgb', 'g', neuron_id, 'rgb', 'g')]['numeric'])
                    b = round(nc['trade:*:{}:{}:{}:{}:{}'.format('has_rgb', 'b', neuron_id, 'rgb', 'b')]['numeric'])
                    opacity = 1.0
                else:
                    r = 255
                    g = 255
                    b = 255
                    opacity = 0.7
                colors.append('rgba({},{},{},{})'.format(r, g, b, opacity))

                labels.append(
                    'NeuroColumn: {}<br>Neuron: {}<br>n_BMU: {}<br>last_bmu: {}<br>n_nn: {}<br>last_nn: {}<br>mean_dist: {}<br>mean_sim: {}<br>r: {}<br>g:{}<br>b:{}<br>community:{}<br>community_prob:{}'.format(
                        coord_id,
                        neuron_id,
                        fabric['neuro_columns'][coord_id]['n_bmu'],
                        fabric['neuro_columns'][coord_id]['last_bmu'],
                        fabric['neuro_columns'][coord_id]['n_nn'],
                        fabric['neuro_columns'][coord_id]['last_nn'],
                        fabric['neuro_columns'][coord_id]['mean_distance'] if 'mean_distance' in fabric['neuro_columns'][coord_id] else None,
                        fabric['neuro_columns'][coord_id]['mean_similarity'] if 'mean_similarity' in fabric['neuro_columns'][coord_id] else None,
                        r, g, b,
                        fabric['neuro_columns'][coord_id]['community_label'],
                        fabric['neuro_columns'][coord_id]['community_label_prob'],
                        ))

                # connect neurons in same column
                #
                if neuron_id < neurons_to_plot[-1]:
                    pair = (min((x, y, neuron_id), (x, y, neuron_id + 1)), max((x, y, neuron_id), (x, y, neuron_id + 1)))

                    pairs.add(pair)

                    x_edge.append(x)
                    x_edge.append(x)
                    x_edge.append(None)

                    y_edge.append(y)
                    y_edge.append(y)
                    y_edge.append(None)

                    nn_z = max(neurons_to_plot) - neuron_id - 1
                    z_edge.append(z)
                    z_edge.append(nn_z)
                    z_edge.append(None)

                # connect neuro_columns
                #
                for nn_id in fabric['neuro_columns'][coord_id]['nn']:
                    nn_x = fabric['neuro_columns'][nn_id]['coord'][0]
                    nn_y = fabric['neuro_columns'][nn_id]['coord'][1]

                    pair = (min((x, y, neuron_id), (nn_x, nn_y, neuron_id)), max((x, y, neuron_id), (nn_x, nn_y, neuron_id)))

                    if pair not in pairs:
                        pairs.add(pair)
                        nn_z = max(neurons_to_plot) - neuron_id
                        x_edge.append(x)
                        x_edge.append(nn_x)
                        x_edge.append(None)

                        y_edge.append(y)
                        y_edge.append(nn_y)
                        y_edge.append(None)

                        z_edge.append(z)
                        z_edge.append(nn_z)
                        z_edge.append(None)

    neuron_scatter = go.Scatter3d(x=x_node, y=y_node, z=z_node, hovertext=labels, mode='markers', marker=dict(size=sizes, color=colors, opacity=1.0))

    amf_edge_scatter = go.Scatter3d(x=x_edge, y=y_edge, z=z_edge, mode='lines', line=dict(width=0.5, color='grey'))

    if nos_winners > 0:
        mean_distance = sum_distance / nos_winners
        mean_mapped = sum_mapped / nos_winners
    else:
        mean_distance = 0.0
        mean_mapped = 0

    fig = go.Figure(data=[neuron_scatter, amf_edge_scatter])
    fig.update_layout(width=1200, height=1200, title=dict(text=title),
                      scene=dict(xaxis_title='X Coord', yaxis_title='Y Coord', zaxis_title='Sequence'))
    print('Nos Mini Columns', len(fabric['neuro_columns']), 'Now Winners:', nos_winners, 'Mean Distance:', mean_distance, 'Mean n_BMU:', mean_mapped)
    fig.show()

def print_neurons_raw_data(raw_data, fabric, title, neuron_ids=None, max_neurons=1):
    raw_x = []
    raw_y = []
    raw_z = []
    raw_colour = []

    for record in raw_data:
        raw_x.append(record['r'])
        raw_y.append(record['g'])
        raw_z.append(record['b'])
        raw_colour.append('rgb({},{},{})'.format(record['r'], record['g'], record['b']))
    raw_scatter = go.Scatter3d(x=raw_x, y=raw_y, z=raw_z, mode='markers+text', marker=dict(size=3, color=raw_colour, opacity=1.0, symbol='square'))

    neuron_x = []
    neuron_y = []
    neuron_z = []
    neuron_colour = []
    neuron_label = []
    neuron_size = []

    if neuron_ids is None:
        neurons = [s for s in range(max_neurons)]
    else:
        neurons = neuron_ids

    for coord in fabric['neuro_columns']:
        for neuron_id in neurons:
            if fabric['neuro_columns'][coord]['n_bmu'] + fabric['neuro_columns'][coord]['n_nn'] > 0 and neuron_id == neurons[-1]:
                neuron_label.append(str(coord))
                if fabric['neuro_columns'][coord]['n_bmu'] > 0:
                    neuron_size.append(fabric['neuro_columns'][coord]['n_bmu'] + 20)
                else:
                    neuron_size.append(fabric['neuro_columns'][coord]['n_nn'] + 10)
            else:
                neuron_label.append(None)
                neuron_size.append(10)

            neuron_x.append(fabric['neuro_columns'][coord]['neuro_column']['trade:*:{}:{}:{}:{}:{}'.format('has_rgb', 'r', neuron_id, 'rgb', 'r')]['numeric'])
            neuron_y.append(fabric['neuro_columns'][coord]['neuro_column']['trade:*:{}:{}:{}:{}:{}'.format('has_rgb', 'g', neuron_id, 'rgb', 'g')]['numeric'])
            neuron_z.append(fabric['neuro_columns'][coord]['neuro_column']['trade:*:{}:{}:{}:{}:{}'.format('has_rgb', 'b', neuron_id, 'rgb', 'b')]['numeric'])
            neuron_colour.append('rgb({},{},{})'.format(int(neuron_x[-1]), int(neuron_y[-1]), int(neuron_z[-1])))

    neuron_scatter = go.Scatter3d(x=neuron_x, y=neuron_y, z=neuron_z, text=neuron_label, mode='markers+text', marker=dict(size=neuron_size, color=neuron_colour, opacity=0.7))

    fig = go.Figure(data=[raw_scatter, neuron_scatter])
    fig.update_layout(scene=dict(xaxis_title='RED', yaxis_title='GREEN', zaxis_title='BLUE'),
                      width=800, height=900,
                      title=dict(text=title))
    fig.show()


def plot_sim_density(fabric_sim):

    min_x = 10000
    min_y = 10000
    max_x = -10000
    max_y = -10000
    for coord in fabric_sim:
        if fabric_sim[coord]['coord'][0] > max_x:
            max_x = fabric_sim[coord]['coord'][0]
        elif fabric_sim[coord]['coord'][0] < min_x:
            min_x = fabric_sim[coord]['coord'][0]
        if fabric_sim[coord]['coord'][1] > max_y:
            max_y = fabric_sim[coord]['coord'][1]
        elif fabric_sim[coord]['coord'][1] < min_y:
            min_y = fabric_sim[coord]['coord'][1]

    z = []
    for y in range(min_y, max_y + 1):
        z.append([0.0] * (max_x - min_x + 1))
        for x in range(min_x, max_x + 1):
            coord = '{}:{}'.format(x, y)
            if coord in fabric_sim:
                z[y + abs(min_y)][x + abs(min_x)] = fabric_sim[coord]['mean_similarity']

    surface = go.Surface(z=z)

    fig = go.Figure(data=[surface])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Similarity'),
                      width=800, height=900)
    fig.show()


def plot_edge_dist(edge_dist):
    dist = []
    max_dist = []
    label = []
    for edge in edge_dist:
        coord = edge[0]
        dist.append(edge[1]['distance'])
        label.append(str(coord))
        """
        if edge[1]['mean_distance'][1] > 1 and edge[1]['mean_distance'][3] > 1:
            max_dist.append(max(((edge[1]['mean_distance'][0] * edge[1]['mean_distance'][1]) - edge[1]['distance'])/(edge[1]['mean_distance'][1] - 1),
                                ((edge[1]['mean_distance'][2] * edge[1]['mean_distance'][3]) - edge[1]['distance']) / (edge[1]['mean_distance'][3] - 1)
                                ))
        else:
        """
        max_dist.append(edge[1]['mean'])

    edge_dist_scatter = go.Scatter(name='motif threshold', text=label, x=[idx for idx in range(len(dist))], y=dist, mode='lines', line=dict(width=1.0, color='red'))
    max_dist_scatter = go.Scatter(name='motif threshold', text=label, x=[idx for idx in range(len(max_dist))], y=max_dist, mode='lines', line=dict(width=1.0, color='blue'))

    fig = go.Figure(data=[edge_dist_scatter, max_dist_scatter])
    fig.update_layout(scene=dict(xaxis_title='edge', yaxis_title='distance'), width=1400, height=900,
                      )
    fig.show()


def test():

    start_time = time.time()

    file_name = '../data/rainbow_trades.json'
    with open(file_name, 'r') as fp:
        raw_data = json.load(fp)

    print('raw data record:', raw_data[:1])

    training_sdrs = {}
    for record in raw_data:
        if record['client'] not in training_sdrs:
            training_sdrs[record['client']] = []
        sdr = SDR()
        sdr.set_item(source_node=('trade', '*'),
                     edge=('generalise', None),
                     target_node=('trade', record['trade_id']),
                     probability=1.0
                     )

        for field in ['r', 'g', 'b']:
            sdr.set_item(source_node=('trade', '*'),
                         edge=('has_rgb', field),
                         target_node=('rgb', field),
                         probability=1.0,
                         numeric=record[field],
                         numeric_min=0,
                         numeric_max=255
                         )

        training_sdrs[record['client']].append((record['trade_id'], sdr))

    short_term_memory = 1
    amf = AMFabric(uid='colours',
                   short_term_memory=short_term_memory,
                   mp_threshold=0.15,
                   structure='star',
                   prune_threshold=0.0,
                   random_seed=221166)

    por_results = []

    client = 'ABC_Ltd'
    s_1 = time.time()
    for rec_id in range(len(training_sdrs[client])):
        por = amf.train(sdr=training_sdrs[client][rec_id][1], ref_id=training_sdrs[client][rec_id][0], non_hebbian_edges=('generalise', ), fast_search=False)
        por_results.append(por)
    e_1 = time.time()

    print('loop', e_1 - s_1)

    fabric = amf.decode_fabric(community_sdr=True)

    plot_fabric_3d(fabric=fabric, title='Trained AMF', neurons_to_plot=[0])

    communities = amf.get_communities()
    print('finished')



if __name__ == '__main__':
    test()
