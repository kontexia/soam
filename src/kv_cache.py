#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from typing import Dict, Set, Any, Optional, Union, List

from src.amgraph import AMFGraph
from src.database_manager_factory import DatabaseManagerFactory

KVCacheValueType = Union[float, int, list, dict, set, str, AMFGraph, None]
""" the types of values that can be stored """


class KVGraphCache:
    def __init__(self, config: Optional[Dict[str, Union[str, int]]] = None) -> None:
        """
        class implements a simple in memory key value cache that can save and restore to a database
        :param config: database keys:\n
                                db_username,
                                db_system,
                                db_name,
                                db_config_file_path
                                db_queries_file_path
                                named_query_collection - postfix for a store's named query collection
        """

        if config is None:
            self._config = {}
        else:
            self._config = config

        if 'default_node_type_edge' not in self._config:
            self._config['default_node_type_edge'] = 'instance_of'
        if 'named_query_collection' not in self._config:
            self._config['named_query_collection'] = 'named_query'
        if 'db_update_batch' not in self._config:
            self._config['db_update_batch'] = 50

        self._store: Dict[str, Dict[str, Any]] = {}
        """ nested dictionary , outer dict keyed on store_names and inner dict implements key value pair """

        self._kv_to_delete: Dict[str, Set[str]] = {}
        """ the store_name (outer dict) and set of keys to delete """

        self._db = DatabaseManagerFactory.get_database_manager(username=self._config['db_username'],
                                                               database_system=config['db_system'],
                                                               database_name=self._config['db_name'],
                                                               config_file_path=self._config['db_config_file_path'],
                                                               queries_file_path=self._config['db_queries_file_path'])

    def persist(self, store_name: Optional[str] = None, keep_cache: bool = True, persist_all_override: bool = False) -> None:
        """
        method to persist contents of cache ont a database
        :param store_name: the store name to persist
        :param keep_cache: if False the store will be deleted after persisting
        :param persist_all_override: if False only data updated will be persisted. If False all data will be persisted
        :return: None
        """

        # get the store_names to persist
        #
        if store_name is None:
            store_names = list(self._store.keys())
        else:
            store_names = [store_name]

        # store_names will also need to include a list of stores to delete
        #
        store_names.extend(list(self._kv_to_delete.keys()))

        # list of stores that will be deleted after being persisted
        #
        stores_to_delete = set()

        # store_name gets mapped to a database collection
        #
        for store in store_names:

            docs_to_insert = {}
            docs_to_delete = {}
            docs_for_name_queries = {}

            if store in self._store:
                # the named query collection for this store
                #
                store_named_query = '{}_{}'.format(store, self._config['named_query_collection'])

                for key in self._store[store]:

                    # persist this key if its been updated (since the last persist) or the override flag has been set
                    #
                    if persist_all_override or self._store[store][key]['updated']:

                        if isinstance(self._store[store][key]['value'], AMFGraph):

                            # for AMFGraphs, key is mapped to named_query
                            #
                            named_query = key

                            # get the nodes and edges from the AMFGraph to persist
                            #
                            nodes_to_persist, edges_to_persist = self._store[store][named_query]['value'].get_data_to_persist(persist_all_override=persist_all_override)

                            for node_type in nodes_to_persist:

                                doc_key = (node_type, 'doc')
                                if doc_key not in docs_to_insert:
                                    docs_to_insert[doc_key] = nodes_to_persist[node_type]
                                else:
                                    docs_to_insert[doc_key].extend(nodes_to_persist[node_type])

                            for edge_type in edges_to_persist:
                                doc_key = (edge_type, 'edge')
                                if doc_key not in docs_to_insert:
                                    docs_to_insert[doc_key] = edges_to_persist[edge_type]
                                else:
                                    docs_to_insert[doc_key].extend(edges_to_persist[edge_type])

                                if edge_type not in docs_for_name_queries:
                                    docs_for_name_queries[edge_type] = {key: set()}

                                if key not in docs_for_name_queries[edge_type]:
                                    docs_for_name_queries[edge_type][key] = set()

                                for edge in edges_to_persist[edge_type]:
                                    docs_for_name_queries[edge_type][key].add(edge['_key'])

                                # also add edge to query collection
                                #
                                query_rec = {'_key': '{}:{}'.format(named_query.replace(' ', '_'), edge_type),
                                             '_named_query': named_query,
                                             '_target_collection': edge_type
                                             }

                                doc_key = (store_named_query, 'doc')
                                if doc_key not in docs_to_insert:
                                    docs_to_insert[doc_key] = [query_rec]
                                else:
                                    docs_to_insert[doc_key].append(query_rec)
                        else:
                            # assume this is a kv
                            #
                            doc_key = (store, 'doc')

                            # each store will map to a collection and have its list of docs to insert
                            #
                            if doc_key not in docs_to_insert:
                                docs_to_insert[doc_key] = []

                            # store value in special _value attribute
                            #
                            docs_to_insert[doc_key].append({'_key': key, '_value': self._store[store][key]['value']})

                        # reset these flags to indicate persistence state
                        #
                        self._store[store][key]['persisted'] = True
                        self._store[store][key]['updated'] = False

            # add in deleted kv's
            #
            if store in self._kv_to_delete:
                if store not in docs_to_delete:
                    docs_to_delete[store] = []

                docs_to_delete[store].extend([{'_key': key} for key in self._kv_to_delete[store]])

                # clear up the _kv_to_delete
                #
                del self._kv_to_delete[store]

            # insert all records
            #
            store_persisted = False
            for doc_key in docs_to_insert:
                store_persisted = True

                # assume doc_key is tuple
                #
                collection_name = doc_key[0]
                collection_type = doc_key[1]
                if not self._db.has_collection(collection_name):

                    # if an edge collection is to be created then set appropriate parameter
                    #
                    if collection_type == 'edge':
                        self._db.create_collection(collection_name=collection_name,
                                                   parameters={'is_edge_collection': True})

                    # else must be normal collection
                    #
                    else:
                        self._db.create_collection(collection_name=collection_name)

                # upsert doc
                #
                self._db.update_many_documents(collection_name=collection_name,
                                               documents=docs_to_insert[doc_key],
                                               parameters={'merge': True}
                                               )

            for collection_name in docs_for_name_queries:
                for graph_key in docs_for_name_queries[collection_name]:

                    # run update_sub_graphs with batches of keys just in case graphs very large
                    #
                    id_list = list(docs_for_name_queries[collection_name][graph_key])

                    try:
                        for idx in range(0, len(id_list), self._config['db_update_batch']):
                            self._db.execute_query('update_sub_graphs', parameters={'collection': {'value': collection_name,
                                                                                                   'type': 'collection'},
                                                                                    'filter_keys': {'value': id_list[idx: idx + self._config['db_update_batch']],
                                                                                                    'type': 'attribute'},
                                                                                    'sub_graph': {'value': '{}_{}'.format(store, graph_key),
                                                                                                  'type': 'attribute'}
                                                                                    })
                    except Exception as e:
                        print('exception thrown', e)

            # delete required records
            #
            for collection_name in docs_to_delete:
                store_persisted = True

                if self._db.has_collection(collection_name):
                    self._db.delete_many_documents(collection_name=collection_name, documents=docs_to_delete[collection_name])

            # maintain list of stores to delete if required
            #
            if not keep_cache and store_persisted and store in self._store:
                stores_to_delete.add(store)

        for store in stores_to_delete:
            del self._store[store]

    def restore(self, store_name: str, key: Optional[Union[List[str], str]] = None, overwrite_exist: bool = True, include_history: bool = False) -> bool:
        """
        method to restore a store_name and optionally specific key
        :param store_name: the store_name to restore
        :param key: a key or list or keys to restore
        :param overwrite_exist: If true overwrite in memory cache with restored data
        :param include_history: If True restore all history
        :return:
        """

        # the collection for named queries
        #
        store_named_query = '{}_{}'.format(store_name, self._config['named_query_collection'])

        # if the named query table exists then must be a graph
        #
        if self._db.has_collection(store_named_query):

            if key is None:
                n_q_collections = self._db.execute_query('get_named_query', parameters={'named_query_col': {'value': store_named_query,
                                                                                                            'type': 'collection'}})

            else:
                if isinstance(key, list):
                    keys = key
                else:
                    keys = [key]

                n_q_collections = self._db.execute_query('get_filtered_named_query', parameters={'named_query_col': {'value': store_named_query,
                                                                                                                     'type': 'collection'},
                                                                                                 'filter_keys': {'value': keys,
                                                                                                                 'type': 'attribute'}})
            node_keys = {}
            for nqe in n_q_collections:

                # add a new graph if it doesnt exist
                #
                if overwrite_exist or store_name not in self._store:
                    self._store[store_name] = {nqe['named_query']: {'value': AMFGraph(), 'persisted': True, 'updated': False}}
                    overwrite_exist = False
                elif nqe['named_query'] not in self._store[store_name]:
                    self._store[store_name][nqe['named_query']] = {'value': AMFGraph(), 'persisted': True, 'updated': False}

                if nqe['named_query'] not in node_keys:
                    node_keys[nqe['named_query']] = {}

                edges_to_restore = []

                sub_graph = '{}_{}'.format(store_name, nqe['named_query'])

                if include_history:
                    edges = self._db.execute_query('get_docs_via_sub_graph', parameters={'collection': {'value': nqe['collection'],
                                                                                                        'type': 'collection'},
                                                                                         'sub_graph': {'value': sub_graph,
                                                                                                       'type': 'attribute'}})
                else:
                    edges = self._db.execute_query('get_docs_via_sub_graph_expiry_ts', parameters={'collection': {'value': nqe['collection'],
                                                                                                                  'type': 'collection'},
                                                                                                   'sub_graph': {'value': sub_graph,
                                                                                                                 'type': 'attribute'},
                                                                                                   'expiry_ts': {'value': None,
                                                                                                                 'type': 'attribute'}})

                for e in edges:
                    edges_to_restore.append(e)
                    if e['_source_type'] not in node_keys[nqe['named_query']]:
                        node_keys[nqe['named_query']][e['_source_type']] = set()

                    if isinstance(e['_source_uid'], str) and ' ' in e['_source_uid']:
                        node_keys[nqe['named_query']][e['_source_type']].add(e['_source_uid'].replace(' ', '_'))
                    else:
                        node_keys[nqe['named_query']][e['_source_type']].add(e['_source_uid'])

                    if e['_target_type'] not in node_keys[nqe['named_query']]:
                        node_keys[nqe['named_query']][e['_target_type']] = set()

                    if isinstance(e['_target_uid'], str) and ' ' in e['_target_uid']:
                        node_keys[nqe['named_query']][e['_target_type']].add(e['_target_uid'].replace(' ', '_'))
                    else:
                        node_keys[nqe['named_query']][e['_target_type']].add(e['_target_uid'])

                self._store[store_name][nqe['named_query']]['value'].restore_edges(edges=edges_to_restore)

            # get nodes attributes for each named query
            #
            for named_query in node_keys:
                for node_collection in node_keys[named_query]:
                    nodes_to_restore = list(self._db.execute_query('get_docs_via_key', parameters={'collection': {'value': node_collection,
                                                                                                                  'type': 'collection'},
                                                                                                   'filter_keys': {'value': list(node_keys[named_query][node_collection]),
                                                                                                                   'type': 'attribute'}}))
                    self._store[store_name][named_query]['value'].restore_nodes(nodes=nodes_to_restore)

        # no named query collection exists so must be simple key value pair
        #
        elif self._db.has_collection(store_name):

            if key is None:

                docs = self._db.execute_query('get_docs_in_collection', parameters={'collection': {'value': store_name,
                                                                                                   'type': 'collection'}})

            else:
                if isinstance(key, list):
                    keys = key
                else:
                    keys = [key]

                docs = self._db.execute_query('get_docs_via_key', parameters={'collection': {'value': store_name,
                                                                                             'type': 'collection'},
                                                                              'filter_keys': {'value': keys,
                                                                                              'type': 'attribute'}})

            for doc in docs:
                if store_name not in self._store:
                    self._store[store_name] = {}

                self._store[store_name][doc['_key']] = {'value': doc['_value'], 'persisted': True, 'updated': False}

        return True

    def set_kv(self, store_name: str, key: str, value: KVCacheValueType, set_update_flag: bool = True) -> bool:
        """
        method to add a key value pair to a store. If key already exists then will overwrite

        :param store_name: the store name
        :param key: the unique key within the store
        :param value: the data to save
        :param set_update_flag: if true then the update flag will be set
        :return: None
        """

        if store_name not in self._store:
            self._store[store_name] = {key: {'value': value,  # the data
                                             'persisted': False,  # indicates if this has been persisted to the database before
                                             'updated': set_update_flag  # indicates if it's been updated and needs persisting
                                             }
                                       }
            stored = True
        elif key not in self._store[store_name]:
            self._store[store_name][key] = {'value': value,  # the data
                                            'persisted': False,  # indicates if this has been persisted to the database before
                                            'updated': set_update_flag  # indicates if it's been updated and needs persisting
                                            }
            stored = True
        else:
            self._store[store_name][key]['value'] = value
            self._store[store_name][key]['persisted'] = False
            self._store[store_name][key]['updated'] = set_update_flag
            stored = True

        return stored

    def del_kv(self, store_name: str, key: Optional[str] = None, delete_from_db: bool = True) -> bool:
        """
        method to delete a key within a store or the whole store. If the store/key has been persisted then the delete will be propagated when persist() is called

        :param store_name: the store name
        :param key: the key to delete
        :param delete_from_db: If true the data will also be deleted from the database
        :return: True if deleted successfully else False
        """
        deleted = False
        if store_name in self._store:

            # need to keep track of all keys deleted for this store
            #
            if delete_from_db and store_name not in self._kv_to_delete:
                self._kv_to_delete[store_name] = set()

            if key is None:
                if delete_from_db:
                    for key in self._store[store_name]:
                        self._kv_to_delete[store_name].add(key)
                del self._store[store_name]
                deleted = True
            elif key in self._store[store_name]:
                if delete_from_db:
                    self._kv_to_delete[store_name].add(key)
                del self._store[store_name][key]
                deleted = True

        return deleted

    def get_kv(self, store_name: str, key: Optional[str] = None) -> KVCacheValueType:
        """
        method to retrieve a stored value
        :param store_name: the store name
        :param key: the key within the store
        :return: KVCacheValueType - either the whole store, the value of the specified key or None if store / key do not exist
        """
        result = None
        if store_name in self._store:
            if key is None:
                result = self._store[store_name]

            elif key in self._store[store_name]:
                result = self._store[store_name][key]['value']

        return result



if __name__ == '__main__':
    import os
    from dotenv import load_dotenv

    g = AMFGraph()

    g.set_edge(source=('Trade', 'XYZ_123'), target=('client', 'abc ltd'), edge=('has', 'client'), prob=1.0)

    load_dotenv()

    config = {'db_name': os.getenv("DB_NAME"),
              'db_username': os.getenv("DB_USERNAME"),
              'db_password': os.getenv("DB_PASSWORD"),
              'db_system': os.getenv("DB_SYSTEM"),
              'db_config_file_path': os.getenv("DB_CONFIG_PATH"),
              'db_queries_file_path': os.getenv("DB_QUERIES_PATH"),
              }

    in_memory_cache = KVGraphCache(config=config)

    in_memory_cache.set_kv(store_name='my_data', key='rec_3', value=g)

    in_memory_cache.persist()

    in_memory_cache_2 = KVGraphCache(config=config)

    in_memory_cache_2.restore(store_name='my_data', key='rec_3')

    restored_graph = in_memory_cache_2.get_kv(store_name='my_data', key='rec_3')

    restored_graph.update_edge(source=('Trade', 'XYZ_123'), target=('client','abc ltd'), edge=('has', 'client'), prob=0.5)

    in_memory_cache_2.set_kv(store_name='my_data', key='rec_3', value=restored_graph)

    in_memory_cache_2.persist()

    print('finished')
