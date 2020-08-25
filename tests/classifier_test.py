#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
import random

from dask.distributed import Client, LocalCluster
import plotly.graph_objects as go

from src.amgraph import AMFGraph
from src.amfabric import AMFabric
from src.distributed_cache import DistributedCache

import json
from dotenv import load_dotenv
import os

import pprint as pp


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


def print_fabric_3d(fabric, title, coords=None, neuron_ids=None, max_neurons=1):
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
    if neuron_ids is not None:
        neurons_to_plot = neuron_ids
    else:
        neurons_to_plot = [n for n in range(max_neurons)]
    if coords is not None:
        coords_to_processes = coords
    else:
        coords_to_processes = list(fabric['neuro_columns'].keys())

    for coord_id in fabric['neuro_columns'].keys():

        if fabric['neuro_columns'][coord_id]['n_bmu'] > 0:
            nos_winners += 1
            sum_distance += fabric['neuro_columns'][coord_id]['mean_distance']
            sum_mapped += fabric['neuro_columns'][coord_id]['n_bmu']

        for neuron_id in neurons_to_plot:
            if 'colour_interest:*:{}:None:{}:{}:{}'.format('has_rgb', neuron_id, 'rgb', 'r') in fabric['neuro_columns'][coord_id]['neuro_column']:

                if neuron_id == neurons_to_plot[0] and fabric['neuro_columns'][coord_id]['n_bmu'] + fabric['neuro_columns'][coord_id]['n_nn'] > 0:
                    sizes.append(20 + fabric['neuro_columns'][coord_id]['n_bmu'])
                else:
                    sizes.append(15)

                labels.append(coord_id + '_' + str(neuron_id))
                nc = fabric['neuro_columns'][coord_id]['neuro_column']

                x = fabric['neuro_columns'][coord_id]['coord'][0]
                y = fabric['neuro_columns'][coord_id]['coord'][1]
                z = max(neurons_to_plot) - neuron_id

                x_node.append(x)
                y_node.append(y)
                z_node.append(z)

                if coord_id in coords_to_processes:
                    r = round(nc['colour_interest:*:{}:None:{}:{}:{}'.format('has_rgb', neuron_id, 'rgb', 'r')]['numeric'])
                    g = round(nc['colour_interest:*:{}:None:{}:{}:{}'.format('has_rgb', neuron_id, 'rgb', 'g')]['numeric'])
                    b = round(nc['colour_interest:*:{}:None:{}:{}:{}'.format('has_rgb', neuron_id, 'rgb', 'b')]['numeric'])
                    opacity = 1.0
                else:
                    r = 255
                    g = 255
                    b = 255
                    opacity = 0.7
                colors.append('rgba({},{},{},{})'.format(r, g, b, opacity))

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

    neuron_scatter = go.Scatter3d(x=x_node, y=y_node, z=z_node, text=labels, mode='markers', marker=dict(size=sizes, color=colors, opacity=1.0))

    amf_edge_scatter = go.Scatter3d(x=x_edge, y=y_edge, z=z_edge, mode='lines', line=dict(width=0.5, color='grey'))

    if nos_winners > 0:
        mean_distance = sum_distance / nos_winners
        mean_mapped = sum_mapped / nos_winners
    else:
        mean_distance = 0.0
        mean_mapped = 0

    fig = go.Figure(data=[neuron_scatter, amf_edge_scatter])
    fig.update_layout(width=1000, height=1000, title=dict(text=title),
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

            neuron_x.append(fabric['neuro_columns'][coord]['neuro_column']['colour_interest:*:{}:None:{}:{}:{}'.format('has_rgb', neuron_id, 'rgb', 'r')]['numeric'])
            neuron_y.append(fabric['neuro_columns'][coord]['neuro_column']['colour_interest:*:{}:None:{}:{}:{}'.format('has_rgb', neuron_id, 'rgb', 'g')]['numeric'])
            neuron_z.append(fabric['neuro_columns'][coord]['neuro_column']['colour_interest:*:{}:None:{}:{}:{}'.format('has_rgb', neuron_id, 'rgb', 'b')]['numeric'])
            neuron_colour.append('rgb({},{},{})'.format(neuron_x[-1], neuron_y[-1], neuron_z[-1]))

    neuron_scatter = go.Scatter3d(x=neuron_x, y=neuron_y, z=neuron_z, text=neuron_label, mode='markers+text', marker=dict(size=neuron_size, color=neuron_colour, opacity=0.7))

    fig = go.Figure(data=[raw_scatter, neuron_scatter])
    fig.update_layout(scene=dict(xaxis_title='RED', yaxis_title='GREEN', zaxis_title='BLUE'),
                      width=800, height=900,
                      title=dict(text=title))
    fig.show()


def test():

    start_time = time.time()

    file_name = '../data/example_colours.json'
    with open(file_name, 'r') as fp:
        raw_data = json.load(fp)

    print('raw data record:', raw_data[:1])

    raw_data_graph = AMFGraph()
    for record in raw_data:
        node_id = ('colour_interest', str(record['record_id']))
        node_attr = {(('has_rgb', None), ('rgb', 'r')): {'prob': 1.0, 'numeric': record['r'], 'numeric_min': 0, 'numeric_max': 255},
                     (('has_rgb', None), ('rgb', 'g')): {'prob': 1.0, 'numeric': record['g'], 'numeric_min': 0, 'numeric_max': 255},
                     (('has_rgb', None), ('rgb', 'b')): {'prob': 1.0, 'numeric': record['b'], 'numeric_min': 0, 'numeric_max': 255},
                     (('has_label', None), ('colour', record['COLOUR'])): {'prob': 1.0},
                     }
        raw_data_graph.set_node(node=node_id, node_attr=node_attr)

    random.seed(291065)

    short_term_memory = 5
    amf = AMFabric(uid='colours',
                   short_term_memory=short_term_memory,
                   mp_threshold=6,
                   structure='star',
                   prune_threshold=0.0)

    por_results = []

    s_1 = time.time()
    for rec_id in range(len(raw_data)):
        sdr = raw_data_graph.get_node_sdr(node=('colour_interest', str(rec_id)), nos_hops=2,
                                          generalised_node_name='*', target_node_edge=('generalise', None))
        por = amf.train(sdr=sdr, ref_id=rec_id)
        por_results.append(por)
    e_1 = time.time()
    print('loop', e_1 - s_1)

    fabric = amf.decode_fabric()
    print_fabric_3d(fabric=fabric, title='Trained AMF', neuron_ids=[0, 1, 2, 3, 4])

    print_fabric_3d(fabric=fabric, title='Trained AMF', neuron_ids=[0, 1, 2, 3, 4], coords=['-1:0'])

    print_neurons_raw_data(raw_data=raw_data, fabric=fabric, title='End of Run', neuron_ids=[0])

    print_anomalies(amf=amf, por_results=por_results)

    context = []
    context_colour_sequence = []

    start_idx = random.randint(0, len(raw_data) - short_term_memory)
    end_idx = start_idx + short_term_memory

    nos_context = 0
    for rec_id in range(start_idx, end_idx):
        context_colour_sequence.append(raw_data[rec_id]['COLOUR'])
        sdr = raw_data_graph.get_node_sdr(node=('colour_interest', str(rec_id)), nos_hops=2, exclude_edges={('has_rgb', None)})

        if nos_context < short_term_memory - 1:

            context.append(sdr)
            nos_context += 1

    print('predict sequence from idx {} given {}'.format(start_idx, context_colour_sequence[:-1]), '\n')
    print('Expect {}'.format(context_colour_sequence[-1]), '\n')

    result = amf.query(sdr=context)

    sdr_result = [(result['neuro_column'][key]['edge_type'],
                   result['neuro_column'][key]['target_type'], result['neuro_column'][key]['target_uid'],
                   result['neuro_column'][key]['neuron_id'],
                   result['neuro_column'][key]['prob'], result['neuro_column'][key]['numeric'] if 'numeric' in result['neuro_column'][key] else None)
                  for key in result['neuro_column']
                  if result['neuro_column'][key]['edge_type'] != 'colour_interest']
    sdr_result.sort(key=lambda x: (x[3], x[0], x[1], -x[4]))
    print(sdr_result)

    end_time = time.time()
    print(end_time - start_time)


    print('finished')



if __name__ == '__main__':
    test()
