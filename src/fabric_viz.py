#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import networkx as nx
import plotly.graph_objects as go


def plot_anomalies(fabric, domain, por_results, title):
    sp_mp = []
    sp_anomaly = []
    sp_motif = []
    sp_anomaly_threshold = []
    sp_mean_mp = []

    sp_anomalies = fabric['domains'][domain]['anomalies']
    sp_motifs = fabric['domains'][domain]['motifs']

    for por in por_results:
        if domain in por:
            sp_mp.append(por[domain]['bmu_distance'])

            sp_anomaly_threshold.append(por[domain]['anomaly_threshold'])
            sp_mean_mp.append(por[domain]['ema_error'])

            if por['ref_id'] in sp_anomalies:
                sp_anomaly.append(sp_anomalies[por['ref_id']]['error'])
            else:
                sp_anomaly.append(None)

            if por['ref_id'] in sp_motifs:
                sp_motif.append(sp_motifs[por['ref_id']]['error'])
            else:
                sp_motif.append(None)

    sp_mp_scatter = go.Scatter(name='Matrix Profile', x=[idx for idx in range(len(sp_mp))], y=sp_mp, mode='lines', line=dict(width=2.0, color='green'))
    sp_anomaly_scatter = go.Scatter(name='Anomaly', x=[idx for idx in range(len(sp_anomaly))], y=sp_anomaly, mode='markers+text', marker=dict(size=10, color='red', opacity=0.7))
    sp_motif_scatter = go.Scatter(name='Motif', x=[idx for idx in range(len(sp_motif))], y=sp_motif, mode='markers+text', marker=dict(size=10, color='green', opacity=0.7))

    sp_meam_mp_scatter = go.Scatter(name='Mean Matrix Profile', x=[idx for idx in range(len(sp_mean_mp))], y=sp_mean_mp, mode='lines', line=dict(width=1.0, color='blue'))
    sp_anomaly_threshold_scatter = go.Scatter(name='Anomaly Threshold', x=[idx for idx in range(len(sp_anomaly_threshold))], y=sp_anomaly_threshold, mode='lines', line=dict(width=1.0, color='red'))

    fig = go.Figure(data=[sp_mp_scatter, sp_anomaly_scatter, sp_motif_scatter, sp_meam_mp_scatter, sp_anomaly_threshold_scatter])
    fig.update_layout(scene=dict(xaxis_title='REF ID', yaxis_title='Error'), width=1400, height=900,
                      title=dict(text=title))
    fig.show()


def plot_fabric_3d(fabric, domain, title, colour_edges=None):

    domain_graph = nx.Graph()

    neurons = fabric['domains'][domain]['neurons']

    communities = {}
    for neuron_id in neurons:
        if neurons[neuron_id]['community'] not in communities:
            communities[neurons[neuron_id]['community']] = {neuron_id}
        else:
            communities[neurons[neuron_id]['community']].add(neuron_id)

    for community in communities:
        if community > -1:
            for neuron_id_1 in communities[community]:
                for neuron_id_2 in communities[community]:
                    if neuron_id_1 != neuron_id_2:
                        domain_graph.add_edge(neuron_id_1, neuron_id_2, weight=1.0)
        else:
            for neuron_id in communities[community]:
                domain_graph.add_node(neuron_id)

    node_layout = nx.spring_layout(domain_graph, dim=3)

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

    for neuron_id in neurons:

        sizes.append(15 + neurons[neuron_id]['n_bmu'])

        x = node_layout[neuron_id][0]
        y = node_layout[neuron_id][1]
        z = node_layout[neuron_id][2]

        x_node.append(x)
        y_node.append(y)
        z_node.append(z)

        sdr = neurons[neuron_id]['generalisation']

        if colour_edges is None:
            colors.append(neurons[neuron_id]['community'])
        else:

            rgbs = {'r': 0, 'g': 0, 'b': 0}
            for edge_key in sdr:
                edge = (sdr[edge_key]['edge_type'], sdr[edge_key]['edge_uid'])
                if edge in colour_edges:
                    rgbs[colour_edges[edge]['rgb']] = (sdr[edge_key]['numeric'] - colour_edges[edge]['min']) / (colour_edges[edge]['min'] - colour_edges[edge]['max'])

            colors.append(f"rgb({rgbs['r']}, {rgbs['g']}, {rgbs['b']})")

        txt = f"Neuron: {neuron_id}<br>n_bmu: {neurons[neuron_id]['n_bmu']}<br>n_runner_up: {neurons[neuron_id]['n_runner_up']}<br>error: {neurons[neuron_id]['ema_error']:.2f}<br>threshold:{neurons[neuron_id]['threshold']:.2f}<br>Community: {neurons[neuron_id]['community']}"
        sdr = neurons[neuron_id]['generalisation']

        edges = [edge for edge in sdr]
        edges.sort(reverse=True)
        for edge in edges:
            txt = f"{txt}<br>{sdr[edge]['edge_type']}: {sdr[edge]['edge_uid']} {sdr[edge]['target_type']}: {sdr[edge]['target_uid']} Prob: {sdr[edge]['prob']:.2f} Numeric: {sdr[edge]['numeric']}"
        labels.append(txt)

        # connect neurons
        #
        synapses = neurons[neuron_id]['synapses']
        for edge in synapses:
            if synapses[edge]['edge_type'] == 'nn':

                nn_x = node_layout[synapses[edge]['target_uid']][0]
                nn_y = node_layout[synapses[edge]['target_uid']][1]
                nn_z = node_layout[synapses[edge]['target_uid']][2]

                pair = (min(neuron_id, synapses[edge]['target_uid']), max(neuron_id, synapses[edge]['target_uid']))

                if pair not in pairs:
                    pairs.add(pair)
                    x_edge.append(x)
                    x_edge.append(nn_x)
                    x_edge.append(None)

                    y_edge.append(y)
                    y_edge.append(nn_y)
                    y_edge.append(None)

                    z_edge.append(z)
                    z_edge.append(nn_z)
                    z_edge.append(None)

    neuron_scatter = go.Scatter3d(x=x_node, y=y_node, z=z_node, hovertext=labels, mode='markers', marker=dict(size=sizes, color=colors, opacity=1.0, colorscale='Rainbow'))

    amf_edge_scatter = go.Scatter3d(x=x_edge, y=y_edge, z=z_edge, mode='lines', line=dict(width=0.5, color='grey'))

    fig = go.Figure(data=[neuron_scatter, amf_edge_scatter])
    fig.update_layout(width=1200, height=1200, title=dict(text=title),
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

    print(f"Domain {domain} Nos Neuron_columns: {len(neurons)} Nos Mapped: {fabric['domains'][domain]['mapped']} Mean Distance: {fabric['domains'][domain]['ema_error']:.2f}")

    fig.show()


# plot the neurons in RGB space
#
def plot_neurons_versus_train_sdr(training_sgs, fabric, xyz_edges, title):
    train_colour = []
    scenes = {'x': 'x',
              'y': 'y',
              'z': 'z'}
    train = {'x': [],
             'y': [],
             'z': []}

    for sg in training_sgs:

        rgbs = {'x': 0, 'y': 0, 'z': 0}
        for edge_key in sg:
            edge = (sg[edge_key]['edge_type'], sg[edge_key]['edge_uid'])
            if edge in xyz_edges:
                train[xyz_edges[edge]['axis']].append(sg[edge_key]['numeric'])
                rgbs[xyz_edges[edge]['axis']] = 255 * (sg[edge_key]['numeric'] - xyz_edges[edge]['min']) / (xyz_edges[edge]['max'] - xyz_edges[edge]['min'])

                scenes[xyz_edges[edge]['axis']] = sg[edge_key]['target_uid']

        if len(xyz_edges) < 3:
            train['z'] = [0.0] * len(train['x'])

        train_colour.append(f"rgb({rgbs['x']}, {rgbs['y']}, {rgbs['z']})")

    train_scatter = go.Scatter3d(x=train['x'], y=train['y'], z=train['z'], mode='markers+text', marker=dict(size=3, color=train_colour, opacity=1.0, symbol='square'))

    nodes = {'x': [],
             'y': [],
             'z': []}
    node_edges = {'x': [],
                  'y': [],
                  'z': []}

    node_colour = []
    node_label = []
    node_size = []

    node_layout = {}
    pairs = set()

    neurons = fabric['domains']['SPATIAL']['neurons']
    for neuron_id in neurons:

        if neuron_id not in node_layout:
            node_layout[neuron_id] = {}

        sdr = neurons[neuron_id]['generalisation']

        rgbs = {'x': 0, 'y': 0, 'z': 0}

        for edge_key in sdr:
            edge = (sdr[edge_key]['edge_type'], sdr[edge_key]['edge_uid'])
            if edge in xyz_edges:
                nodes[xyz_edges[edge]['axis']].append(sdr[edge_key]['numeric'])
                node_layout[neuron_id][xyz_edges[edge]['axis']] = sdr[edge_key]['numeric']
                rgbs[xyz_edges[edge]['axis']] = 255 * (sdr[edge_key]['numeric'] - xyz_edges[edge]['min']) / (xyz_edges[edge]['max'] - xyz_edges[edge]['min'])

        if len(xyz_edges) < 3:
            nodes['z'] = [0.0] * len(nodes['x'])
            node_layout[neuron_id]['z'] = 0.0
            rgbs['z'] = 0

        node_colour.append(f"rgb({rgbs['x']}, {rgbs['y']}, {rgbs['z']})")

        node_size.append(15 + neurons[neuron_id]['n_bmu'])

        txt = f"Neuron: {neuron_id}<br>n_bmu: {neurons[neuron_id]['n_bmu']}<br>n_runner_up: {neurons[neuron_id]['n_runner_up']}<br>error: {neurons[neuron_id]['ema_error']:.2f}<br>threshold:{neurons[neuron_id]['threshold']:.2f}<br>Community: {neurons[neuron_id]['community']}"

        edges = [edge for edge in sdr]
        edges.sort(reverse=True)
        for edge in edges:
            txt = f"{txt}<br>{sdr[edge]['edge_type']}: {sdr[edge]['edge_uid']} {sdr[edge]['target_type']}: {sdr[edge]['target_uid']} Prob: {sdr[edge]['prob']:.2f} Numeric: {sdr[edge]['numeric']}"
        node_label.append(txt)

    for neuron_id in neurons:

        # connect neurons
        #
        synapses = neurons[neuron_id]['synapses']
        for edge in synapses:
            if synapses[edge]['edge_type'] == 'nn':

                nn_id = synapses[edge]['target_uid']

                pair = (min(neuron_id, nn_id), max(neuron_id, nn_id))

                if pair not in pairs:
                    pairs.add(pair)
                    node_edges['x'].append(node_layout[neuron_id]['x'])
                    node_edges['x'].append(node_layout[nn_id]['x'])
                    node_edges['x'].append(None)

                    node_edges['y'].append(node_layout[neuron_id]['y'])
                    node_edges['y'].append(node_layout[nn_id]['y'])
                    node_edges['y'].append(None)

                    node_edges['z'].append(node_layout[neuron_id]['z'])
                    node_edges['z'].append(node_layout[nn_id]['z'])
                    node_edges['z'].append(None)

    neuron_scatter = go.Scatter3d(x=nodes['x'], y=nodes['y'], z=nodes['z'], hovertext=node_label, mode='markers+text', marker=dict(size=node_size, color=node_colour, opacity=0.7))
    edge_scatter = go.Scatter3d(x=node_edges['x'], y=node_edges['y'], z=node_edges['z'], mode='lines', line=dict(width=0.5, color='grey'))

    fig = go.Figure(data=[train_scatter, edge_scatter, neuron_scatter])

    fig.update_layout(scene=dict(xaxis_title=scenes['x'], yaxis_title=scenes['y'], zaxis_title=scenes['z']),
                      width=1200, height=1200,
                      title=dict(text=title))
    fig.show()
