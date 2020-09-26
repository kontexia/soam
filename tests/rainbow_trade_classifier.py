#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
import random
from copy import deepcopy

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


# plot the fabric structure
#
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
    nos_bmus = 0

    if neurons_to_plot is None:
        neurons_to_plot = [n for n in range(short_term_memory)]

    if coords_to_highlight is None:
        coords_to_highlight = list(fabric['neuro_columns'].keys())

    for coord_id in fabric['neuro_columns'].keys():

        if fabric['neuro_columns'][coord_id]['n_bmu'] > 0:
            nos_bmus += 1

        for neuron_id in neurons_to_plot:

            # if there are edges for this neuron_id
            #
            if sum([1 for edge in fabric['neuro_columns'][coord_id]['neuro_column'] if fabric['neuro_columns'][coord_id]['neuro_column'][edge]['neuron_id'] == neuron_id]) > 0:

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

                txt = 'NeuroColumn: {}<br>Neuron: {}<br>n_BMU: {}<br>last_bmu: {}<br>n_nn: {}<br>last_nn: {}<br>mean_dist: {:.2f}<br>mean_sim: {:.2f}<br>community:{}'.format(coord_id,
                                                                                                                                                                              neuron_id,
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord_id]['n_bmu'],
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord_id]['last_bmu'],
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord_id]['n_nn'],
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord_id]['last_nn'],
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord_id][
                                                                                                                                                                                  'mean_distance'] if 'mean_distance' in
                                                                                                                                                                                                      fabric[
                                                                                                                                                                                                          'neuro_columns'][
                                                                                                                                                                                                          coord_id] else None,
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord_id][
                                                                                                                                                                                  'mean_similarity'] if 'mean_similarity' in
                                                                                                                                                                                                        fabric[
                                                                                                                                                                                                            'neuro_columns'][
                                                                                                                                                                                                            coord_id] else None,
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord_id][
                                                                                                                                                                                  'community'],
                                                                                                                                                                              )
                edges = list(fabric['neuro_columns'][coord_id]['neuro_column'].keys())
                edges.sort()
                for edge in edges:
                    if fabric['neuro_columns'][coord_id]['neuro_column'][edge]['neuron_id'] == neuron_id:

                        # get the numeric if it exists
                        #
                        if 'numeric' in fabric['neuro_columns'][coord_id]['neuro_column'][edge]:
                            numeric = '{:.2f}'.format(fabric['neuro_columns'][coord_id]['neuro_column'][edge]['numeric'])
                        else:
                            numeric = None

                        txt = '{}<br>{}: {} {}: {} Prob: {:.2f} Numeric: {}'.format(txt,
                                                                                    fabric['neuro_columns'][coord_id]['neuro_column'][edge]['edge_type'],
                                                                                    fabric['neuro_columns'][coord_id]['neuro_column'][edge]['edge_uid'],
                                                                                    fabric['neuro_columns'][coord_id]['neuro_column'][edge]['target_type'],
                                                                                    fabric['neuro_columns'][coord_id]['neuro_column'][edge]['target_uid'],
                                                                                    fabric['neuro_columns'][coord_id]['neuro_column'][edge]['prob'],
                                                                                    numeric
                                                                                    )
                labels.append(txt)

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

    fig = go.Figure(data=[neuron_scatter, amf_edge_scatter])
    fig.update_layout(width=1200, height=1200, title=dict(text=title),
                      scene=dict(xaxis_title='X Coord', yaxis_title='Y Coord', zaxis_title='Sequence'))

    print('Nos Neuron_columns: {} Nos BMUs: {}, Nos Mapped: {} Mean Distance: {:.2f} Mean similarity: {:.2f}'.format(len(fabric['neuro_columns']),
                                                                                                                     nos_bmus,
                                                                                                                     fabric['mapped'],
                                                                                                                     fabric['mean_distance'],
                                                                                                                     fabric['mean_similarity']))

    fig.show()


# plot the neurons in RGB space
#
def plot_neurons_trades(trades, fabric, title, neurons_to_plot=None, short_term_memory=1):
    raw_x = []
    raw_y = []
    raw_z = []
    raw_colour = []

    for record in trades:
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
    hover_text = []

    if neurons_to_plot is None:
        neurons_to_plot = [s for s in range(short_term_memory)]

    for coord in fabric['neuro_columns']:
        for neuron_id in neurons_to_plot:

            # plot the neuron if it exists
            if sum([1 for edge in fabric['neuro_columns'][coord]['neuro_column'] if fabric['neuro_columns'][coord]['neuro_column'][edge]['neuron_id'] == neuron_id]) > 0:

                if fabric['neuro_columns'][coord]['n_bmu'] + fabric['neuro_columns'][coord]['n_nn'] > 0 and neuron_id == neurons_to_plot[-1]:

                    neuron_label.append(str(coord))
                    if fabric['neuro_columns'][coord]['n_bmu'] > 0:
                        neuron_size.append(fabric['neuro_columns'][coord]['n_bmu'] + 20)
                    else:
                        neuron_size.append(fabric['neuro_columns'][coord]['n_nn'] + 15)

                else:
                    neuron_label.append(None)
                    neuron_size.append(15)

                neuron_x.append(fabric['neuro_columns'][coord]['neuro_column']['trade:*:{}:{}:{}:{}:{}'.format('has_rgb', 'r', neuron_id, 'rgb', 'r')]['numeric'])
                neuron_y.append(fabric['neuro_columns'][coord]['neuro_column']['trade:*:{}:{}:{}:{}:{}'.format('has_rgb', 'g', neuron_id, 'rgb', 'g')]['numeric'])
                neuron_z.append(fabric['neuro_columns'][coord]['neuro_column']['trade:*:{}:{}:{}:{}:{}'.format('has_rgb', 'b', neuron_id, 'rgb', 'b')]['numeric'])
                neuron_colour.append('rgb({},{},{})'.format(round(neuron_x[-1]), round(neuron_y[-1]), round(neuron_z[-1])))

                txt = 'NeuroColumn: {}<br>Neuron: {}<br>n_BMU: {}<br>last_bmu: {}<br>n_nn: {}<br>last_nn: {}<br>mean_dist: {:.2f}<br>mean_sim: {:.2f}<br>community:{}'.format(coord,
                                                                                                                                                                              neuron_id,
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord]['n_bmu'],
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord]['last_bmu'],
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord]['n_nn'],
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord]['last_nn'],
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord][
                                                                                                                                                                                  'mean_distance'] if 'mean_distance' in
                                                                                                                                                                                                      fabric[
                                                                                                                                                                                                          'neuro_columns'][
                                                                                                                                                                                                          coord] else None,
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord][
                                                                                                                                                                                  'mean_similarity'] if 'mean_similarity' in
                                                                                                                                                                                                        fabric[
                                                                                                                                                                                                            'neuro_columns'][
                                                                                                                                                                                                            coord] else None,
                                                                                                                                                                              fabric['neuro_columns'][
                                                                                                                                                                                  coord]['community'],
                                                                                                                                                                              )
                edges = list(fabric['neuro_columns'][coord]['neuro_column'].keys())
                edges.sort()
                for edge in edges:
                    if fabric['neuro_columns'][coord]['neuro_column'][edge]['neuron_id'] == neuron_id:

                        # get the numeric if it exists
                        #
                        if 'numeric' in fabric['neuro_columns'][coord]['neuro_column'][edge]:
                            numeric = '{:.2f}'.format(fabric['neuro_columns'][coord]['neuro_column'][edge]['numeric'])
                        else:
                            numeric = None

                        txt = '{}<br>{}: {} {}: {} Prob: {:.2f} Numeric: {}'.format(txt,
                                                                                    fabric['neuro_columns'][coord]['neuro_column'][edge]['edge_type'],
                                                                                    fabric['neuro_columns'][coord]['neuro_column'][edge]['edge_uid'],
                                                                                    fabric['neuro_columns'][coord]['neuro_column'][edge]['target_type'],
                                                                                    fabric['neuro_columns'][coord]['neuro_column'][edge]['target_uid'],
                                                                                    fabric['neuro_columns'][coord]['neuro_column'][edge]['prob'],
                                                                                    numeric
                                                                                    )
                hover_text.append(txt)

    neuron_scatter = go.Scatter3d(x=neuron_x, y=neuron_y, z=neuron_z, hovertext=hover_text, text=neuron_label, mode='markers+text', marker=dict(size=neuron_size, color=neuron_colour, opacity=0.7))

    fig = go.Figure(data=[raw_scatter, neuron_scatter])
    fig.update_layout(scene=dict(xaxis_title='RED', yaxis_title='GREEN', zaxis_title='BLUE'),
                      width=1200, height=1200,
                      title=dict(text=title))
    fig.show()


# plot matrix profile
#
def plot_matrix_profile(amf, por_results, std_factor=1):
    sp_mp = []
    sp_anomaly = []
    sp_motif = []
    sp_anomaly_threshold = []
    sp_motif_threshold = []
    sp_mean = []
    sp_std_upper = []
    sp_std_lower = []

    sp_anomalies = amf.get_anomalies()
    sp_motifs = amf.get_motifs()

    for por in por_results:
        sp_mp.append(por['bmu_distance'])

        sp_mean.append(por['mean_distance'])
        sp_std_upper.append(por['mean_distance'] + (std_factor * por['std_distance']))
        sp_std_lower.append(max(por['mean_distance'] - (std_factor * por['std_distance']), 0.0))

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

    sp_motif_threshold_scatter = go.Scatter(name='motif threshold', x=[idx for idx in range(len(sp_motif_threshold))], y=sp_motif_threshold, mode='lines', line=dict(width=1.0, color='blue'))
    sp_anomaly_threshold_scatter = go.Scatter(name='anomaly threshold', x=[idx for idx in range(len(sp_anomaly_threshold))], y=sp_anomaly_threshold, mode='lines', line=dict(width=1.0, color='red'))

    sp_mean_scatter = go.Scatter(name='mean', x=[idx for idx in range(len(sp_mean))], y=sp_mean, mode='lines', line=dict(width=2.0, color='black'))
    sp_std_upper_scatter = go.Scatter(name='stdev', x=[idx for idx in range(len(sp_std_upper))], y=sp_std_upper, mode='lines', line=dict(width=2.0, color='yellow'))
    sp_std_lower_scatter = go.Scatter(name='stdev', x=[idx for idx in range(len(sp_std_lower))], y=sp_std_lower, mode='lines', line=dict(width=2.0, color='yellow'))

    fig = go.Figure(data=[sp_mp_scatter, sp_anomaly_scatter, sp_motif_scatter, sp_motif_threshold_scatter, sp_anomaly_threshold_scatter, sp_mean_scatter, sp_std_upper_scatter, sp_std_lower_scatter, ])
    fig.update_layout(scene=dict(xaxis_title='REF ID', yaxis_title='Similarity'), width=1400, height=900,
                      title=dict(text='AMFabric Pooler Anomalies & Motifs'))
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
                     target_node=('trade', str(record['trade_id'])),
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
                   random_seed=221166,
                   min_cluster_size=2,
                   cluster_start_threshold=0.5,
                   cluster_step=0.01
                   )

    por_results = []

    client = 'ABC_Ltd'
    s_1 = time.time()
    for rec_id in range(len(training_sdrs[client])):
        por = amf.train(sdr=training_sdrs[client][rec_id][1], ref_id=training_sdrs[client][rec_id][0], non_hebbian_edges=('generalise', ), fast_search=False)
        por_results.append(por)
    e_1 = time.time()

    print('loop', e_1 - s_1)

    fabric = amf.decode_fabric(all_details=True)

    plot_fabric_3d(fabric=fabric, title='Trained AMF', neurons_to_plot=[0])

    plot_matrix_profile(amf, por_results, std_factor=1)

    print('finished')




if __name__ == '__main__':

    test()
