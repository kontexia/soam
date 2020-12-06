#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from dotenv import load_dotenv
from src.subgraph import  SubGraph
from src.pubsub import PubSub





if __name__ == '__main__':

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

    load_dotenv()

    config = {'db_name': os.getenv("DB_NAME"),
              'db_username': os.getenv("DB_USERNAME"),
              'db_password': os.getenv("DB_PASSWORD"),
              'db_system': os.getenv("DB_SYSTEM"),
              'db_config_file_path': os.getenv("DB_CONFIG_PATH"),
              'db_queries_file_path': os.getenv("DB_QUERIES_PATH"),
              'scheduler_address': os.getenv("DASK_SCHEDULER")}

    publisher = PubSub(uid='train_publisher', config=config)

    messages = []
    for ref_id in range(len(training_data)):
        msg = {'ref_id': ref_id,
               'fabric_uid': 'test_cluster',
               'request': 'TEMPORAL',
               'sub_graph': training_data[ref_id][0],
               'search_edge_types': {'has_signal'},
               'learn_edge_types': {'has_signal'},
               'normalise_groups': {'has_signal': 'has_signal',
                                    'has_trg': 'has_trg',
                                    'has_label': 'has_label'}
               }
        messages.append(msg)

    publisher.publish(topic='TRAINING_DATA',
                      msg=messages,
                      persist=False)

    print('got here')
    publisher.publish(topic='CLOSE_DOWN', msg={}, persist=False)
    print('finished')


