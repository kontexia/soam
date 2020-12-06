from src.subgraph import SubGraph
import src.fabric_viz as viz
import src.neuro_gas_fabric as ng
from sklearn.datasets import make_moons, make_swiss_roll

import time
import json


def moon_test(generalise=1.0):

    data_set, labels = make_moons(n_samples=200, noise=0.05,
                                  random_state=0)

    training_sgs = {}
    for idx in range(len(data_set)):
        label_sg = SubGraph()

        label_sg.set_item(source_node=('datum', '*'),
                          edge=('has_label', None),
                          target_node=('label', str(labels[idx])),
                          probability=1.0)

        context_sdr = SubGraph()

        key = context_sdr.set_item(source_node=('datum', '*'),
                                   edge=('has_x', None),
                                   target_node=('data_type', 'x'),
                                   probability=1.0,
                                   numeric=data_set[idx][0],
                                   )

        key = context_sdr.set_item(source_node=('datum', '*'),
                                   edge=('has_y', None),
                                   target_node=('data_type', 'y'),
                                   probability=1.0,
                                   numeric=data_set[idx][1],
                                   )

        training_sgs[idx] = (context_sdr, label_sg)

    fabric = ng.create_fabric(fabric_uid='test',
                              anomaly_threshold=4.0,
                              slow_learn_rate=0.1,
                              fast_learn_rate=0.7,
                              ltm_attention_heads=3,
                              prune_threshold=0.01,
                              generalisation_sensitivity=generalise,
                              normalise_groups={'has_x': 'has_x',
                                                'has_y': 'has_y'})

    pors = []
    start_time = time.time()
    for ref_id in range(len(training_sgs)):

        por = ng.learn_spatial(fabric=fabric,
                               ref_id=str(ref_id),
                               sub_graph=training_sgs[ref_id][0],
                               search_edge_types={'has_x', 'has_y'},
                               learn_edge_types={'has_x', 'has_y'})

        classify_por = ng.learn_association(fabric=fabric,
                                            ref_id=str(ref_id),
                                            domain='CLASSIFICATION',
                                            sub_graph=training_sgs[ref_id][1],
                                            search_edge_types={'has_label'},
                                            learn_edge_types={'has_label'},
                                            learn_nearest_neighbours=False)
        por.update(classify_por)

        pors.append(por)
    end_time = time.time()

    print(f'avg time {(end_time-start_time)/len(training_sgs)} secs')

    decoded_fabric = ng.decode_fabric(fabric)

    viz.plot_fabric_3d(fabric=decoded_fabric,
                       domain='SPATIAL',
                       title='Moon')

    viz.plot_neurons_versus_train_sg(training_sgs=[training_sgs[i][0] for i in training_sgs],
                                      fabric=decoded_fabric,
                                      xyz_edges={('has_x', None): {'axis': 'x',
                                                                   'min': -3,
                                                                   'max': 3},
                                                 ('has_y', None): {'axis': 'y',
                                                                   'min': -3,
                                                                   'max': 3}},
                                      title='Moon')
    viz.plot_anomalies(fabric=decoded_fabric, domain='SPATIAL', por_results=pors, title="Moon")

    print('finished')


def swiss_roll_test(generalise=1.0):

    data_set, labels = make_swiss_roll(n_samples=400, noise=0.05, random_state=0,)

    training_sgs = {}
    for idx in range(len(data_set)):
        label_sg = SubGraph()

        label_sg.set_item(source_node=('datum', '*'),
                          edge=('has_label', None),
                          target_node=('label', str(labels[idx])),
                          probability=1.0)

        context_sdr = SubGraph()

        key = context_sdr.set_item(source_node=('datum', '*'),
                                   edge=('has_x', None),
                                   target_node=('data_type', 'x'),
                                   probability=1.0,
                                   numeric=data_set[idx][0],
                                   )

        key = context_sdr.set_item(source_node=('datum', '*'),
                                   edge=('has_y', None),
                                   target_node=('data_type', 'y'),
                                   probability=1.0,
                                   numeric=data_set[idx][1],
                                   )

        key = context_sdr.set_item(source_node=('datum', '*'),
                                   edge=('has_z', None),
                                   target_node=('data_type', 'z'),
                                   probability=1.0,
                                   numeric=data_set[idx][2],
                                   )

        training_sgs[idx] = (context_sdr, label_sg)

    fabric = ng.create_fabric(fabric_uid='test',
                              anomaly_threshold=4.0,
                              slow_learn_rate=0.1,
                              fast_learn_rate=0.7,
                              ltm_attention_heads=3,
                              prune_threshold=0.01,
                              generalisation_sensitivity=generalise,
                              normalise_groups={'has_x': 'has',
                                                'has_y': 'has',
                                                'has_z': 'has',
                                                'has_label': 'has_label'
                                                })

    pors = []
    start_time = time.time()
    for ref_id in range(len(training_sgs)):

        por = ng.learn_spatial(fabric=fabric,
                               ref_id=str(ref_id),
                               sub_graph=training_sgs[ref_id][0],
                               search_edge_types={'has_x', 'has_y', 'has_z'},
                               learn_edge_types={'has_x', 'has_y', 'has_z'})

        classify_por = ng.learn_association(fabric=fabric,
                                            ref_id=str(ref_id),
                                            domain='CLASSIFICATION',
                                            sub_graph=training_sgs[ref_id][1],
                                            search_edge_types={'has_label'},
                                            learn_edge_types={'has_label'},
                                            learn_nearest_neighbours=False)
        por.update(classify_por)

        pors.append(por)
    end_time = time.time()

    print(f'avg time {(end_time-start_time)/len(training_sgs)} secs')

    decoded_fabric = ng.decode_fabric(fabric)

    viz.plot_fabric_3d(fabric=decoded_fabric,
                       domain='SPATIAL',
                       title='Swiss Roll')

    viz.plot_neurons_versus_train_sg(training_sgs=[training_sgs[i][0] for i in training_sgs],
                                      fabric=decoded_fabric,
                                      xyz_edges={('has_x', None): {'axis': 'x',
                                                                   'min': -12,
                                                                   'max': 22},
                                                 ('has_y', None): {'axis': 'y',
                                                                   'min': -12,
                                                                   'max': 22},
                                                 ('has_z', None): {'axis': 'z',
                                                                   'min': -12,
                                                                   'max': 22}},
                                      title='Swiss Roll')

    viz.plot_anomalies(fabric=decoded_fabric, domain='SPATIAL', por_results=pors, title="Swiss Roll")

    print('finished')


def colour_test(generalise=1.0):
    start_time = time.time()

    file_name = '../data/rainbow_trades.json'
    with open(file_name, 'r') as fp:
        raw_data = json.load(fp)

    print('raw data record:', raw_data[:1])

    training_sgs = {}
    for record in raw_data:
        if record['client'] not in training_sgs:
            training_sgs[record['client']] = []
        sg = SubGraph()
        sg.set_item(source_node=('trade', '*'),
                    edge=('has_trade', None),
                    target_node=('trade', str(record['trade_id'])),
                    probability=1.0
                    )

        for field in ['r', 'g', 'b']:
            key = sg.set_item(source_node=('trade', '*'),
                              edge=('has_rgb', field),
                              target_node=('rgb', field),
                              probability=1.0,
                              numeric=record[field],
                              )
    
        label_sg = SubGraph()
        label_sg.set_item(source_node=('trade', '*'),
                          edge=('has_colour', None),
                          target_node=('colour', record['label']),
                          probability=1.0
                          )

        training_sgs[record['client']].append((record['trade_id'], sg, label_sg, record))

    fabrics = {}

    pors = {}
    for client in training_sgs:
        if client not in fabrics:
            fabrics[client] = ng.create_fabric(fabric_uid=client,
                                               anomaly_threshold=4.0,
                                               slow_learn_rate=0.1,
                                               fast_learn_rate=0.7,
                                               ltm_attention_heads=3,
                                               prune_threshold=0.01,
                                               normalise_groups={'has_rgb': 'has_rgb',
                                                                 'has_trade': 'has_trade',
                                                                 'has_colour': 'has_colour',
                                                                 })
            pors[client] = []

        start_time = time.time()
        for record in training_sgs[client]:
            ref_id = record[0]
            spatial_sg = record[1]
            label_sg = record[2]
            por = ng.learn_spatial(fabric=fabrics[client],
                                   ref_id=str(ref_id),
                                   sub_graph=spatial_sg,
                                   search_edge_types={'has_rgb'},
                                   learn_edge_types={'has_rgb'})

            temporal_por = ng.learn_temporal(fabric=fabrics[client],
                                             ref_id=str(ref_id),
                                             sub_graph=spatial_sg,
                                             search_edge_types={'has_rgb'},
                                             learn_edge_types={'has_rgb'})
            por.update(temporal_por)

            classify_por = ng.learn_association(fabric=fabrics[client],
                                                ref_id=str(ref_id),
                                                domain='CLASSIFICATION',
                                                sub_graph=label_sg,
                                                search_edge_types={'has_colour'},
                                                learn_edge_types={'has_colour'},
                                                learn_nearest_neighbours=False)
            por.update(classify_por)

            pors[client].append(por)

        end_time = time.time()

        print(f'{client} avg time {(end_time - start_time) / len(training_sgs[client])} secs')

        decoded_fabric = ng.decode_fabric(fabrics[client])

        viz.plot_fabric_3d(fabric=decoded_fabric,
                           domain='SPATIAL',
                           title=client + ' SPATIAL')

        viz.plot_neurons_versus_train_sg(training_sgs=[record[1] for record in training_sgs[client]],
                                         fabric=decoded_fabric,
                                         xyz_edges={('has_rgb', 'r'): {'axis': 'x',
                                                                       'min': 0,
                                                                       'max': 255},
                                                    ('has_rgb', 'g'): {'axis': 'y',
                                                                       'min': 0,
                                                                       'max': 255},
                                                    ('has_rgb', 'b'): {'axis': 'z',
                                                                       'min': 0,
                                                                       'max': 255}},
                                         title=client + ' SPATIAL')

        viz.plot_anomalies(fabric=decoded_fabric, domain='SPATIAL', por_results=pors[client], title=client + " SPATIAL")
        viz.plot_anomalies(fabric=decoded_fabric, domain='TEMPORAL', por_results=pors[client], title=client + " TEMPORAL")

    print('finished')


def square_test(generalise=1.0):
    import time

    training_data = []

    for cycles in range(8):
        for low_value in range(3):
            signal_sg = SubGraph()
            signal_sg.set_item(source_node=('timeseries', 'ts'),
                               edge=('has_signal', None),
                               target_node=('signal', 'value'),
                               probability=1.0,
                               numeric=0.0)
            label_sg = SubGraph()
            label_sg.set_item(source_node=('timeseries', 'ts'),
                              edge=('has_class', None),
                              target_node=('class', 'low'),
                              probability=1.0)

            target_sg = SubGraph()
            target_sg.set_item(source_node=('timeseries', 'ts'),
                               edge=('has_trg', None),
                               target_node=('trg', 'value'),
                               probability=1.0,
                               numeric=10.0)

            training_data.append((signal_sg, label_sg, target_sg))

        for high_value in range(3):
            signal_sg = SubGraph()
            signal_sg.set_item(source_node=('timeseries', 'ts'),
                               edge=('has_signal', None),
                               target_node=('signal', 'value'),
                               probability=1.0,
                               numeric=1.0, )
            label_sg = SubGraph()
            label_sg.set_item(source_node=('timeseries', 'ts'),
                              edge=('has_label', None),
                              target_node=('label', 'High'),
                              probability=1.0)

            target_sg = SubGraph()
            target_sg.set_item(source_node=('timeseries', 'ts'),
                               edge=('has_trg', None),
                               target_node=('trg', 'value'),
                               probability=1.0,
                               numeric=20.0)

            training_data.append((signal_sg, label_sg, target_sg))

    fabric = ng.create_fabric(fabric_uid='test',
                              anomaly_threshold=4.0,
                              slow_learn_rate=0.1,
                              fast_learn_rate=0.7,
                              ltm_attention_heads=3,
                              prune_threshold=0.01,
                              normalise_groups={'has_signal': 'has_signal',
                                                'has_trg': 'has_trg',
                                                'has_label': 'has_label'})

    pors = []
    start_time = time.time()
    for ref_id in range(len(training_data)):
        por = ng.learn_spatial(fabric=fabric,
                               ref_id=str(ref_id),
                               sub_graph=training_data[ref_id][0],
                               search_edge_types={'has_signal'},
                               learn_edge_types={'has_signal'})

        temporal_por = ng.learn_temporal(fabric=fabric,
                                         ref_id=str(ref_id),
                                         sub_graph=training_data[ref_id][0],
                                         search_edge_types={'has_signal'},
                                         learn_edge_types={'has_signal'})
        por.update(temporal_por)

        classify_por = ng.learn_association(fabric=fabric,
                                            ref_id=str(ref_id),
                                            domain='CLASSIFICATION',
                                            sub_graph=training_data[ref_id][1],
                                            search_edge_types={'has_label'},
                                            learn_edge_types={'has_label'},
                                            learn_nearest_neighbours=False)
        por.update(classify_por)

        predict_por = ng.learn_association(fabric=fabric,
                                           ref_id=str(ref_id),
                                           domain='PREDICT',
                                           sub_graph=training_data[ref_id][2],
                                           search_edge_types={'has_trg'},
                                           learn_edge_types={'has_trg'},
                                           learn_nearest_neighbours=True)
        por.update(predict_por)

        pors.append(por)
    end_time = time.time()

    print(f'avg time {(end_time - start_time) / len(training_data)} secs')

    decoded_fabric = ng.decode_fabric(fabric)

    query_result = ng.next_in_sequence(fabric=fabric)

    print('finished')


if __name__ == '__main__':
    #moon_test(generalise=1.0)
    #swiss_roll_test(generalise=1.0)
    colour_test(generalise=1.0)
    #square_test(generalise=1.0)

