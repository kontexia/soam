
from abc import ABC, abstractmethod
import json


class DatabaseManager(ABC):
    database_queries = None

    @property
    @abstractmethod
    def querying_language_dictionary(self):
        raise NotImplementedError

    ##########
    # database functions
    ##########
    def delete_database(self, database_name):
        raise NotImplementedError

    @abstractmethod
    def create_database(self, database_name, parameters=None):
        raise NotImplementedError

    ##########
    # graph functions
    ##########

    @abstractmethod
    def has_graph(self, graph_name, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def get_graph(self, graph_name, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def create_graph(self, graph_name, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def delete_graph(self, graph_name, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def has_edge_definition(self, graph_name, edge_collection_name, parameters=None):
        # somewhat specific to ArangoDB....
        raise NotImplementedError

    @abstractmethod
    def create_edge_definition(self, graph_name, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def delete_edge_definition(self, graph_name, edge_collection_name, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def replace_edge_definition(self, graph_name, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def graph_dict_to_db(self, graph_dict, collection_mode='update'):
        raise NotImplementedError

    def parse_sub_graph(self, sub_graph, db_vertices, db_edges, db_edge_definitions, is_vertex_of_edges,
                        is_edge_of_edges, parent_type=None, parent_key=None, is_root=True):
        try:
            # print("- Parsing from: " + sub_graph['_key']+" - parent_type: "+str(parent_type)+" - parent_key: "+str(parent_key))
            node_type = sub_graph['_type']

            if node_type not in db_vertices:
                db_vertices[node_type] = {}
            dict = {}
            # if is_root:
            #    dict['am_graph_dict'] = sub_graph

            for key in sub_graph.keys():
                if key != 'sub_graph':
                    dict[key] = sub_graph[key]
            # dict['id'] = tuple(dict['id'])

            if dict['_key'] not in db_vertices[node_type].keys():
                db_vertices[node_type][dict['_key']] = dict

            if not is_root:
                is_vertex_of_edge_doc = {}
                _from_key = sub_graph['_key']
                _to_key = parent_key
                is_vertex_of_edge_doc['_from'] = node_type + '/' + _from_key
                is_vertex_of_edge_doc['_to'] = parent_type + '/' + _to_key
                is_vertex_of_edge_doc['_key'] = node_type + '_' + _from_key + '_' + parent_type + '_' + _to_key
                is_vertex_of_edges.append(is_vertex_of_edge_doc)

            if 'sub_graph' in sub_graph.keys():
                if len(sub_graph['sub_graph']['nodes']) > 0:
                    for node in sub_graph['sub_graph']['nodes']:
                        self.parse_sub_graph(node,
                                             db_vertices,
                                             db_edges,
                                             db_edge_definitions,
                                             is_vertex_of_edges,
                                             is_edge_of_edges,
                                             parent_type=sub_graph['_type'],
                                             parent_key=sub_graph['_key'],
                                             is_root=False)

                db_vertices_by_node_id = {}
                for type in db_vertices.keys():
                    # for vertex in db_vertices[type]:
                    db_vertices_by_node_id.update(db_vertices[type])

                for edge in sub_graph['sub_graph']['links']:
                    # print("- Parsing edge: ", edge)

                    edge_type = edge['_type']
                    if edge_type not in db_edges:
                        db_edges[edge_type] = {}
                        db_edge_definitions[edge_type] = {'from_vertex_collections': [],
                                                          'to_vertex_collections': []}

                    # fix code to get type from nodes dictionary
                    # _from_key = self.tuple_to_id(edge['source'])
                    # _to_key = self.tuple_to_id(edge['target'])
                    _from_key = edge['source_ref']
                    _to_key = edge['target_ref']

                    _from_type = db_vertices_by_node_id[_from_key]['_type']
                    _to_type = db_vertices_by_node_id[_to_key]['_type']

                    edge['_from'] = _from_type + '/' + _from_key
                    edge['_to'] = _to_type + '/' + _to_key
                    db_edge_definitions[edge_type]['from_vertex_collections'].append(_from_type)
                    db_edge_definitions[edge_type]['to_vertex_collections'].append(_to_type)

                    if edge['_key'] not in db_edges[edge_type].keys():
                        db_edges[edge_type][edge['_key']] = edge

                    is_edge_of_edge_doc = {}
                    _from_key = edge['_key']
                    _to_key = parent_key
                    is_edge_of_edge_doc['_from'] = edge_type + '/' + _from_key
                    is_edge_of_edge_doc['_to'] = sub_graph['_type'] + '/' + sub_graph['_key']
                    is_edge_of_edge_doc['_key'] = edge_type + ':' + _from_key + '::' + sub_graph['_type'] + ':' + \
                                                  sub_graph['_key']

                    # sub_graph['_type'] if is_root else parent_type + '_' + \
                    # sub_graph['_key'] if is_root else _to_key
                    # pp.pprint(is_edge_of_edge_doc)

                    is_edge_of_edges.append(is_edge_of_edge_doc)
            else:
                # print("- Return from: " + sub_graph['_key'] + " - parent_type: " + str(
                #    parent_type) + " - parent_key: " + str(parent_key))
                return
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
            print("ArangoDBDatabaseManager.parse_sub_graph - raised Excetion: " + str(e))
            print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")

    def db_to_graph_dict(self, root_type, root_key):
        if False:
            cursor = self.execute_query('get_vertex_by_id',
                                        parameters={'collection_name': {'value': root_type,
                                                                        'type': 'collection'},
                                                    'vertex_id': {'value': root_type + '/' + root_key,
                                                                  'type': 'attribute'}})
            for item in cursor:
                return item['am_graph_dict']
        else:
            try:
                cursor = self.execute_query('get_vertex_by_id',
                                            parameters={'collection_name': {'value': root_type,
                                                                            'type': 'collection'},
                                                        'vertex_id': {'value': root_type + '/' + root_key,
                                                                      'type': 'attribute'}})

                root_dict = {}
                for item in cursor:
                    root_dict = item

                    cursor = self.execute_query('get_neighbour_vertices_via_is_vertex_of_edges',
                                                parameters={'start_node_id': {'value': root_type + '/' + root_key,
                                                                              'type': 'attribute'}})
                    if cursor.batch().__len__() > 0:
                        root_dict['sub_graph'] = {'nodes': [],
                                                  'links': []}
                        for item in cursor:
                            # print(item['_key'])
                            root_dict['sub_graph']['nodes'].append(
                                self.db_to_graph_dict(item['_type'], item['_key']))

                        for node in root_dict['sub_graph']['nodes']:
                            node['id'] = tuple(i for i in node['id'])
                            node.pop('_rev')
                            node.pop('_id')

                        cursor_edge = self.execute_query('get_neighbour_edges_via_is_edge_of',
                                                         parameters={'start_node_id': {
                                                             'value': root_type + '/' + root_key,
                                                             'type': 'attribute'}})
                        if cursor_edge.batch().__len__() > 0:
                            keys_exclusion_list = ['_id', '_rev', '_from', '_to']
                            for item_edge in cursor_edge:
                                edge_dict = {}

                                for key in item_edge.keys():
                                    if key not in keys_exclusion_list:
                                        edge_dict[key] = item_edge[key]

                                edge_dict['id'] = tuple(i for i in edge_dict['source'])
                                edge_dict['source'] = tuple(i for i in edge_dict['source'])
                                edge_dict['target'] = tuple(i for i in edge_dict['target'])
                                root_dict['sub_graph']['links'].append(edge_dict)

                return root_dict

            except Exception as e:
                print("! ! ! ! ! ! ! ! ! ! ")
                print("DatabaseManager.db_to_graph_dict - raised an Exception: " + str(e))
                print("! ! ! ! ! ! ! ! ! ! ")

    def integrate_db_collection(self, collection_name, db_collection, collection_mode, is_edge_collection=False):
        preexisting_documents = self.has_many_documents(collection_name, db_collection)
        new_documents = [doc for doc in db_collection if doc not in preexisting_documents]

        if preexisting_documents:
            if collection_mode == 'update':
                self.update_many_documents(collection_name, db_collection)
            else:
                self.replace_many_documents(collection_name, db_collection)
        if new_documents:
            self.insert_many_documents(collection_name, new_documents,
                                       parameters={'is_edge_collection': is_edge_collection})

    ##########
    # collection functions
    ##########

    @abstractmethod
    def has_collection(self, collection_name, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def get_collection(self, collection_name, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def get_collections(self):
        raise NotImplementedError

    @abstractmethod
    def get_edge_collections(self):
        raise NotImplementedError

    @abstractmethod
    def get_collection_as_a_list(self, collection_name):
        raise NotImplementedError

    @abstractmethod
    def get_collection_as_a_dict(self, collection_name):
        raise NotImplementedError

    def get_create_collection(self, collection_name, parameters=None):
        try:
            if self.has_collection(collection_name):
                collection = self.get_collection(collection_name)
            else:
                collection = self.create_collection(collection_name, parameters)
            return collection
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("DatabaseManager.get_create_collection - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    @abstractmethod
    def create_collection(self, collection_name, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def delete_collection(self, collection_name, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def truncate_collection(self, collection_name, parameters=None):
        raise NotImplementedError

    ##########
    # document functions
    ##########

    @abstractmethod
    def has_document(self, collection_name, key, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def has_many_documents(self, collection_name, documents, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def get_document(self, collection_name, key, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def insert_document(self, collection_name, document, parameters=None):
        raise NotImplementedError

    def check_insert_document(self, collection_name, document, parameters=None):
        try:
            if not self.has_document(collection_name, document['_key']):
                self.insert_document(collection_name, document, parameters)
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("DatabaseManager.check_insert_document - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def check_insert_document_batch(self, collection, document, parameters=None):
        if document['_key'] not in collection.keys():
            collection[document['_key']] = document

    def check_insert_many_documents(self, collection_name, documents, parameters=None):
        try:
            print("DatabaseManager.check_insert_many_documents - started...")
            new_docs = [doc for doc in documents if doc not in self.has_many_documents(collection_name, documents)]
            self.insert_many_documents(collection_name, new_docs, parameters=parameters)
            print("DatabaseManager.check_insert_many_documents - end")
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("DatabaseManager.check_insert_many_documents - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    @abstractmethod
    def insert_many_documents(self, collection_name, documents, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def replace_document(self, collection_name, document, parameters=None):
        raise NotImplementedError

    def replace_document_batch(self, collection, document):
        collection[document['_key']] = document

    @abstractmethod
    def delete_document(self, collection_name, document, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def delete_many_documents(self, collection_name, documents, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def replace_many_documents(self, collection_name, documents, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def update_document(self, collection_name, document, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def update_many_documents(self, collection_name, documents, parameters=None):
        raise NotImplementedError

    def upsert_document(self, collection_name, document, parameters=None):
        try:
            if self.has_collection(collection_name):
                if self.has_document(collection_name, document['_key']):
                    self.update_document(collection_name, document)
                else:
                    self.insert_document(collection_name, document)
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("DatabaseManager.upsert_document - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def upsert_increment_document_batch(self, collection, document, key_field='_key', incremented_field='weight'):
        if document[key_field] in collection.keys():
            if incremented_field in document.keys():
                collection[document[key_field]][incremented_field] += 1
        else:
            collection[document[key_field]] = document

    ##########
    # query functions
    ##########

    @abstractmethod
    def get_query(self, query_name):
        raise NotImplementedError

    @abstractmethod
    def get_query_result(self, query, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def execute_query(self, query_name, parameters=None, filters=None):
        raise NotImplementedError

    @abstractmethod
    def get_dictionary_from_cursor(self, cursor, key_attribute):
        raise NotImplementedError

    @abstractmethod
    def translate_filters(self, filters):
        raise NotImplementedError

    @abstractmethod
    def translate_to_querying_language(self, input):
        raise NotImplementedError

    def get_database_queries(self, database_queries_file_path=None):
        if database_queries_file_path:
            with open(database_queries_file_path, 'r') as readfile:
                self.database_queries = json.load(readfile)
            return self.database_queries
        else:
            return self.database_queries

