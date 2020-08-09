from src.neuro_column import SDR
from src.kv_cache import KVGraphCache
from src.amgraph import AMFGraph


def test_cache():
    config = {'db_name': 'AMF',
              'db_username': 'stephen',
              'db_password': 'kontexia.io',
              'db_system': 'arango_db',
              'db_config_file_path': '~/kontexia/dev/amf/soam/databases_configuration.json',
              'db_queries_file_path': '~/kontexia/dev/amf/soam/database_queries.json', }

    cache = KVGraphCache(config=config)

    cache.set_kv('simple_store', key='my_key_1', value={'name': 'stephen', 'age': 21})

    cache.persist()


    new_cache = KVGraphCache(config=config)

    new_cache.restore(store_name='simple_store')

    my_graph = AMFGraph()

    node_attr = {('has_interest', ('interest', 'banking'), 0): 0.5,
                 ('has_interest', ('interest', 'real_estate'), 0): 0.5,
                 ('has_classification', ('classification', 'low_touch'), 0): 123.0
                 }

    node_properties = {'id': 123}

    edge_properties = {'update_id': 123}

    my_graph.set_node(node=('client', 'a1'), node_attr=node_attr, node_prop=node_properties, edge_prop=edge_properties, timestamp=None)

    print(my_graph)

    cache.set_kv(store_name='graph_store', key='graph_1', value=my_graph)

    cache.persist()

    new_cache.restore(store_name='graph_store', key='graph_1')

    print('finished')


def test_sdr():
    sdr = SDR()
    print(sdr)

if __name__ == '__main__':

    test_cache()
    test_sdr()
    print('finished')