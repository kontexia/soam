#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.distributed_amfabric import DistributedAMFabric
from src.sdr import SDR
import json
import os
from dotenv import load_dotenv


def train_fabric():
    # setup configuration
    #
    load_dotenv()
    config = {'db_name': os.getenv("DB_NAME"),
              'db_username': os.getenv("DB_USERNAME"),
              'db_password': os.getenv("DB_PASSWORD"),
              'db_system': os.getenv("DB_SYSTEM"),
              'db_config_file_path': os.getenv("DB_CONFIG_PATH"),
              'db_queries_file_path': os.getenv("DB_QUERIES_PATH"),
              'fabric_graph': 'HSG',
              'scheduler_address': os.getenv("DASK_SCHEDULER")}

    # make a connection to the distributed amfabric
    #
    amf = DistributedAMFabric(config=config)

    # read in some trades
    #
    file_name = '../data/rainbow_trades.json'
    with open(file_name, 'r') as fp:
        raw_data = json.load(fp)

    # create SDRs for the trades
    #
    training_sdrs = {}
    for record in raw_data:
        if record['client'] not in training_sdrs:
            training_sdrs[record['client']] = []

        sdr = SDR()

        # connect the generic trade to the trade id
        #
        sdr.set_item(source_node=('trade', '*'),
                     edge=('generalise', None),
                     target_node=('trade', str(record['trade_id'])),
                     probability=1.0
                     )

        # connect the generic trade to the trade id
        #
        sdr.set_item(source_node=('trade', '*'),
                     edge=('has_colour', 'colour'),
                     target_node=('colour', str(record['label'])),
                     probability=1.0
                     )

        # connect the generic trade to the three rgb attributes
        #
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

    # select a client
    #
    client = 'ABC_Ltd'

    # ensure the distributed amfabric area exists for this client
    #
    amf.create_distributed_fabric(fabric_uid=client,
                                  short_term_memory=1,
                                  mp_threshold=0.15,
                                  structure='star',
                                  prune_threshold=0.001,
                                  min_cluster_size=2,
                                  cluster_start_threshold=0.5,
                                  cluster_step=0.01,
                                  random_seed=221166,
                                  restore=True)

    por_results = []

    # for each SDR trade executed by this client train the distributed amfabric area
    #
    for rec_id in range(len(training_sdrs[client])):
        por = amf.train(fabric_uid=client,
                        sdr=training_sdrs[client][rec_id][1],
                        ref_id=training_sdrs[client][rec_id][0],
                        non_hebbian_edges=('generalise', ),
                        fast_search=False)
        por_results.append(por)

    p_results = []
    for por in por_results:
        p_results.append(por.result())

    # and persist
    #
    result = amf.persist_fabric(fabric_uid=client)
    result = result.result()
    print('finished')


if __name__ == '__main__':

    train_fabric()

