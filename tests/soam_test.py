from src.neuro_column import SDR
from src.kv_cache import KVGraphCache
from src.distributed_cache import DistributedCache
from src.amgraph import AMFGraph
from src.distributed_amfabric import DistributedAMFabric
import json
from src.amfabric import AMFabric
from dotenv import load_dotenv
import os

def test_cache():

    file_name = '../data/example_colours.json'
    with open(file_name, 'r') as fp:
        raw_data = json.load(fp)

    print('raw data record:', raw_data[:1])

    raw_data_graph = AMFGraph()

    for record in raw_data:
        # each record represents an interest in colour so create a 'colour_interest' node
        #
        node_id = ('colour_interest', str(record['record_id']))

        # each colour_interest node has four attribute nodes
        #
        node_attr = {('has_rgb', ('rgb', 'r')): {'prob': 1.0, 'numeric': record['r'], 'numeric_min': 0, 'numeric_max': 1.0},
                     ('has_rgb', ('rgb', 'g')): {'prob': 1.0, 'numeric': record['g'], 'numeric_min': 0, 'numeric_max': 1.0},
                     ('has_rgb', ('rgb', 'b')): {'prob': 1.0, 'numeric': record['b'], 'numeric_min': 0, 'numeric_max': 1.0},
                     ('has_label', ('colour', record['COLOUR'])): {'prob': 1.0},
                     }
        raw_data_graph.set_node(node=node_id, node_attr=node_attr)

    load_dotenv()

    config = {'db_name': os.getenv("DB_NAME"),
              'db_username': os.getenv("DB_USERNAME"),
              'db_password': os.getenv("DB_PASSWORD"),
              'db_system': os.getenv("DB_SYSTEM"),
              'db_config_file_path': os.getenv("DB_CONFIG_PATH"),
              'db_queries_file_path': os.getenv("DB_QUERIES_PATH"),
              'scheduler_address': "tcp://192.168.1.67:8786"}

    cache = KVGraphCache(config=config)
    #cache = DistributedCache(config=config)

    cache.set_kv('interest_graphs', key='raw_colour_interest', value=raw_data_graph)

    cache.persist()

    #new_cache = KVGraphCache(config=config)

    #new_cache.restore(store_name='interest_graphs')

    print('finished')

def test_amfabric():

    load_dotenv()

    config = {'db_name': os.getenv("DB_NAME"),
              'db_username': os.getenv("DB_USERNAME"),
              'db_password': os.getenv("DB_PASSWORD"),
              'db_system': os.getenv("DB_SYSTEM"),
              'db_config_file_path': os.getenv("DB_CONFIG_PATH"),
              'db_queries_file_path': os.getenv("DB_QUERIES_PATH"),
              'scheduler_address': "tcp://192.168.1.67:8786"}

    cache = KVGraphCache(config=config)

    cache.restore(store_name='HSG', key='AMFabric_test')

    rpg = cache.get_kv(store_name='HSG', key='AMFabric_test')

    amf = AMFabric(uid='test',
                   short_term_memory=2,
                   mp_threshold=6,
                   structure='star',
                   prune_threshold=0.001,
                   random_seed=221166)

    if rpg is not None:
        amf.set_persist_graph(pg=rpg)

    sdr_1 = SDR()
    sdr_1.set_item(source_node=('client', 'x'), edge=('has', 'y'), target_node=('nominal_amount', 'nominal'), probability=1.0, numeric=100, numeric_min=0, numeric_max=200)

    search_por = amf.search_for_bmu(sdr=sdr_1, ref_id=1, non_hebbian_edges=None)

    learn_por = amf.learn(search_por=search_por)

    pg = amf.get_persist_graph(ref_id=1)

    cache.set_kv('HSG', key='AMFabric_test', value=pg)

    cache.persist()

    cache.restore('HSG', key='AMFabric_test')

    pg = cache.get_kv('HSG', key='AMFabric_test')

    amf.set_persist_graph(pg=pg)

    sdr_2 = SDR()
    sdr_2.set_item(source_node=('client', 'x'), edge=('has', 'y'), target_node=('nominal_amount', 'nominal'), probability=1.0, numeric=150, numeric_min=0, numeric_max=200)

    search_por = amf.search_for_bmu(sdr=sdr_2, ref_id=2, non_hebbian_edges=None)

    learn_por = amf.learn(search_por=search_por)

    pg = amf.get_persist_graph(pg_to_update=pg)

    cache.set_kv('HSG', key='AMFabric_test', value=pg)

    cache.persist()




    print('finished')


def test_dist_amfabric():
    load_dotenv()

    config = {'db_name': os.getenv("DB_NAME"),
              'db_username': os.getenv("DB_USERNAME"),
              'db_password': os.getenv("DB_PASSWORD"),
              'db_system': os.getenv("DB_SYSTEM"),
              'db_config_file_path': os.getenv("DB_CONFIG_PATH"),
              'db_queries_file_path': os.getenv("DB_QUERIES_PATH"),
              'scheduler_address': "tcp://192.168.1.67:8786",
              'fabric_uid': 'test',
              'fabric_short_term_memory': 2,
              'fabric_mp_threshold': 6,
              'fabric_structure': 'star',
              'fabric_prune_threshold': 0.001,
              'fabric_random_seed': 221166
              }

    amf = DistributedAMFabric(config=config)
    sdr_1 = SDR()
    sdr_1.set_item(source_node=('client', 'x'), edge=('has', 'y'), target_node=('nominal_amount', 'nominal'), probability=1.0, numeric=100, numeric_min=0, numeric_max=200)

    search_por = amf.search_for_bmu(sdr=sdr_1, ref_id=1, non_hebbian_edges=None)

    learn_por = amf.learn(search_por=search_por.result())

    sdr_2 = SDR()
    sdr_2.set_item(source_node=('client', 'x'), edge=('has', 'y'), target_node=('nominal_amount', 'nominal'), probability=1.0, numeric=150, numeric_min=0, numeric_max=200)

    search_por = amf.search_for_bmu(sdr=sdr_2, ref_id=2, non_hebbian_edges=None)

    learn_por = amf.learn(search_por=search_por.result())

    print('finished')


def test_sdr():
    sdr = SDR()
    print(sdr)

if __name__ == '__main__':

    test_dist_amfabric()

    #test_amfabric()
    #test_cache()

    print('finished')
