#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from dotenv import load_dotenv
from src.subgraph import SubGraph
from dask.distributed import Client
import src.neuro_gas_fabric as ng
from src.pubsub import PubSub
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import queue
from dask import bag as db
import jsonpickle as jp
import json
import os

import logging
import time


class DistributedNeuroGas:
    def __init__(self, uid, config):

        self.uid = uid

        self.config = config

        self.fabrics = {}

        self.dask_scheduler = Client(config['scheduler_address'])

        self.subscriber = PubSub(uid=uid, config=config)

        self.close_down = False

        self.pool = ThreadPoolExecutor(config['max_number_threads'])
        self.futures = []

    def process_fabric(self, fabric_uid):

        logging.debug(f'processing fabric requests for {fabric_uid}')

        # the filename to restore and persist the fabric
        #
        fabric_filename = os.path.join(config['fabric_persist_path'], fabric_uid) + '.json'
        por_filename = os.path.join(config['fabric_persist_path'], fabric_uid) + '_por.json'

        t_time = time.time()
        n_requests = 0
        try:

            while True:
                request = self.fabrics[fabric_uid]['queue'].get(timeout=0)
                n_requests += 1

                # if the fabric doesnt exist then retrieve from persistence store or create new
                #
                if 'fabric' not in self.fabrics[fabric_uid]:

                    start_time = time.time()
                    if not os.path.exists(fabric_filename):
                        if 'normalise_groups' in request:
                            normalise_groups = request['normalise_groups']
                        else:
                            normalise_groups = None

                        self.fabrics[fabric_uid]['fabric'] = ng.create_fabric(fabric_uid=fabric_uid, normalise_groups=normalise_groups)
                    else:
                        with open(fabric_filename, 'r') as jfile:
                            fabric_pickle = json.load(jfile)
                            self.fabrics[fabric_uid]['fabric'] = jp.decode(fabric_pickle)

                    end_time = time.time()
                    logging.debug(f'restored fabric {fabric_uid} within {end_time-start_time} secs')

                por = {}

                # process SPATIAL and TEMPORAL requests
                #
                if request['request'] in ['SPATIAL', 'TEMPORAL']:
                    por = ng.learn_spatial(fabric=self.fabrics[fabric_uid]['fabric'],
                                           ref_id=request['ref_id'],
                                           sub_graph=request['sub_graph'],
                                           search_edge_types=request['search_edge_types'],
                                           learn_edge_types=request['learn_edge_types']
                                           )

                    if request['request'] == 'TEMPORAL':
                        temporal_por = ng.learn_temporal(fabric=self.fabrics[fabric_uid]['fabric'],
                                                         ref_id=request['ref_id'],
                                                         sub_graph=request['sub_graph'],
                                                         search_edge_types=request['search_edge_types'],
                                                         learn_edge_types=request['learn_edge_types']
                                                         )
                        por.update(temporal_por)

                elif request['request'] == 'ASSOCIATION':

                    # get the matching spatial neuron_id to apply the association to
                    #
                    spatial_query = ng.query_domain(fabric=self.fabrics[fabric_uid]['fabric'],
                                                    domain='SPATIAL',
                                                    sub_graph=request['spatial_sub_graph'],
                                                    association_depth=0)
                    por['spatial_query'] = spatial_query

                    spatial_bmu_id = spatial_query['bmu_id']

                    # learn the association
                    #
                    association_por = ng.learn_association(fabric=self.fabrics[fabric_uid]['fabric'],
                                                           ref_id=request['ref_id'],
                                                           domain=request['association_domain'],
                                                           spatial_neuron_id=spatial_bmu_id,
                                                           sub_graph=request['association_sub_graph'],
                                                           search_edge_types=request['search_edge_types'],
                                                           learn_edge_types=request['learn_edge_types'],
                                                           learn_nearest_neighbours=request['learn_nearest_neighbours'])
                    por.update(association_por)

                logging.debug(f'saving por for {fabric_uid}')

                por_pickle = jp.encode(por)
                if not os.path.exists(por_filename):
                    with open(por_filename, 'wt') as jfile:
                        json.dump(por_pickle, jfile)
                else:
                    with open(por_filename, 'at') as jfile:
                        json.dump(por_pickle, jfile)

        except queue.Empty:
            pass

        if fabric_uid in self.fabrics and 'fabric' in self.fabrics[fabric_uid]:

            e_time = time.time()

            if n_requests > 0:
                logging.debug(f'processed {n_requests} avg per request {(e_time-t_time)/n_requests} secs now persisting fabric {fabric_uid}')

            # persist
            #
            fabric_pickle = jp.encode(self.fabrics[fabric_uid]['fabric'])
            with open(fabric_filename, 'w') as jfile:
                json.dump(fabric_pickle, jfile)

            logging.debug(f'persisting fabric {fabric_uid}')

        return fabric_uid

    def process_training_msg(self, msg):

        logging.debug('processing message')

        requests = msg['msg']
        for request in requests:
            if request['fabric_uid'] not in self.fabrics:
                self.fabrics[request['fabric_uid']] = {'queue': queue.Queue()}
                self.futures.append(self.pool.submit(self.process_fabric, request['fabric_uid']))
            else:
                self.fabrics[request['fabric_uid']]['queue'].put(request)

        # clear up any futures that are done
        #
        for future in self.futures:
            if future.done():
                fabric_uid = future.result()
                del self.fabrics[fabric_uid]
                self.futures.remove(future)

    def process_close_down_msg(self, msg):

        while len(self.futures) > 0:
            for future in self.futures:
                if future.done():
                    fabric_uid = future.result()
                    del self.fabrics[fabric_uid]
                    self.futures.remove(future)

        self.close_down = True

    def run(self):
        self.subscriber.subscribe(topic=config['training_topic'], callback=self.process_training_msg)
        self.subscriber.subscribe(topic=config['close_down_topic'], callback=self.process_close_down_msg)

        while not self.close_down:
            self.subscriber.listen(timeout=0.1, persist=False)

        logging.debug('closing down')


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    load_dotenv()

    config = {'db_name': os.getenv("DB_NAME"),
              'db_username': os.getenv("DB_USERNAME"),
              'db_password': os.getenv("DB_PASSWORD"),
              'db_system': os.getenv("DB_SYSTEM"),
              'db_config_file_path': os.getenv("DB_CONFIG_PATH"),
              'db_queries_file_path': os.getenv("DB_QUERIES_PATH"),
              'scheduler_address': os.getenv("DASK_SCHEDULER"),
              'training_topic': 'TRAINING_DATA',
              'close_down_topic': 'CLOSE_DOWN',
              'max_number_threads': 100,
              'fabric_persist_path': '/users/stephen/kontexia/dev/soam/data'
              }

    dng = DistributedNeuroGas(uid='DNG', config=config)
    dng.run()
    print('finished')

