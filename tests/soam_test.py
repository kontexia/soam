from src.neuro_column import SDR
from src.kv_cache import KVGraphCache
from src.distributed_cache import DistributedCache
from src.amgraph import AMFGraph
import json
from src.amfabric import AMFabric

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

    config = {'db_name': 'AMF',
              'db_username': 'stephen',
              'db_password': 'kontexia.io',
              'db_system': 'arango_db',
              'db_config_file_path': '~/kontexia/dev/amf/soam/databases_configuration.json',
              'db_queries_file_path': '~/kontexia/dev/amf/soam/database_queries.json',
              'scheduler_address': 'localhost:8786'}

    cache = KVGraphCache(config=config)
    #cache = DistributedCache(config=config)

    cache.set_kv('interest_graphs', key='raw_colour_interest', value=raw_data_graph)

    cache.persist()

    #new_cache = KVGraphCache(config=config)

    #new_cache.restore(store_name='interest_graphs')

    print('finished')

def test_amfabric():

    amf = AMFabric(uid='colours',
                   short_term_memory=2,
                   mp_threshold=6,
                   structure='star',
                   prune_threshold=0.001)

    sdr_1 = SDR()
    sdr_1.set_item(source_node=('client', 'x'), edge=('has', 'y'), target_node=('nominal_amount', 'nominal'),probability=1.0,numeric=100, numeric_min=0, numeric_max=200)

    search_por = amf.search_for_bmu(sdr=sdr_1, ref_id=1, non_hebbian_edges=None)

    learn_por = amf.learn(search_por=search_por)

    pg = amf.get_persist_graph()


    sdr_2 = SDR()
    sdr_2.set_item(source_node=('client', 'x'), edge=('has', 'y'), target_node=('nominal_amount', 'nominal'), probability=1.0, numeric=150, numeric_min=0, numeric_max=200)


    search_por = amf.search_for_bmu(sdr=sdr_1, ref_id=1, non_hebbian_edges=None)

    learn_por = amf.learn(search_por=search_por)

    pg = amf.get_persist_graph(pg_to_update=pg)
    print('finished')




def test_sdr():
    sdr = SDR()
    print(sdr)

if __name__ == '__main__':

    test_amfabric()
    #test_cache()

    print('finished')
