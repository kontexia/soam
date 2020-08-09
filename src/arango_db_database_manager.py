from src.database_manager import DatabaseManager
from arango import ArangoClient
import json


class ArangoDBDatabaseManager(DatabaseManager):
    is_in_memory = False

    database_URL = None
    database_name = None
    username = None
    password = None
    database = None  #
    query_language = None  #
    database_queries = None  #

    client = None

    querying_language_dictionary = {'=': '==',
                                    '!=': '!=',
                                    '>': '>',
                                    '>=': '>=',
                                    '<': '<',
                                    '<=': '<=',
                                    'and': '&&',
                                    'or': '||',
                                    'not': '!',
                                    'in': 'in',
                                    'not in': 'not in',
                                    'like': 'like',
                                    'not like': 'not like',
                                    '=~': '=~',
                                    '!~': '!~',
                                    'limit': 'limit'}

    def __init__(self, parameters=None):

        self.is_in_memory = parameters.get('is_in_memory', False)
        self.database_system = parameters.get('database_system', 'arango_db')
        self.database_name = parameters.get('database_name', '_system')
        self.database_queries = self.get_database_queries(parameters.get('database_queries_file_path'))
        self.database_connection_parameters = self.get_database_connection_parameters(
            parameters.get('databases_configuration_file_path'),
            self.database_system,
            self.database_name)

        if self.database_connection_parameters:
            self.database_URL = self.database_connection_parameters['database_URL']
            self.username = self.database_connection_parameters['username']
            self.password = self.database_connection_parameters['password']
            self.query_language = self.database_connection_parameters['query_language']
            self.create_client()

    ##########
    # database functions
    ##########

    def get_database_connection_parameters(self,
                                           databases_configuration_file_path=None,
                                           database_system='arango_db',
                                           database_name='_system'):
        try:
            with open(databases_configuration_file_path, 'r') as readfile:
                databases_configuration = json.load(readfile)
            return databases_configuration[database_system][database_name]
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.get_database_connection_parameters - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def create_client(self):
        try:
            self.client = ArangoClient(hosts=self.database_URL)
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.create_client - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def create_database_connection(self):
        try:
            self.database = self.client.db(self.database_name, username=self.username, password=self.password)
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.create_database_connection - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def test_database_connection(self):
        if self.database is None:
            self.create_database_connection()

    def delete_database(self, parameters=None):
        try:
            self.test_database_connection()
            if self.database is None:
                print("ArangoDBDatabaseManager.delete_database - no connection to database: ", self.database_name)
            else:
                for collection in self.database.collections():
                    if not collection['system']:
                        self.database.delete_collection(collection['name'])

        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.delete_database - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def create_database(self, database_name, parameters=None):
        try:
            self.test_database_connection()
            self.database.create_database(database_name, users=parameters.get('users') if parameters else None)
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.create_database - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    ##########
    # graph functions
    ##########

    def has_graph(self, graph_name, parameters=None):
        try:
            return self.database.has_graph(graph_name)
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.has_graph - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def get_graph(self, graph_name, parameters=None):
        try:
            self.test_database_connection()
            return self.database.graph(graph_name)
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.get_graph - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def create_graph(self, graph_name, parameters=None):
        try:
            self.test_database_connection()
            graph = self.database.create_graph(graph_name)
            return graph

        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.create_graph - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def delete_graph(self, graph_name, parameters=None):
        try:
            self.test_database_connection()
            self.database.delete_graph(graph_name)
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.delete_graph - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def has_edge_definition(self, graph_name, edge_collection_name, parameters=None):
        try:
            self.test_database_connection()

            if self.has_graph(graph_name):
                graph = self.get_graph(graph_name)
                return graph.has_edge_definition(edge_collection_name)
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.has_edge_definition - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def create_edge_definition(self, graph_name, parameters=None):
        try:
            self.test_database_connection()

            if self.has_graph(graph_name):
                graph = self.get_graph(graph_name)
                edge_collection_name = parameters.get('edge_collection_name')

                if not graph.has_edge_definition(edge_collection_name):
                    from_vertex_collection_names_list = parameters.get('from_vertex_collection_names_list')
                    to_vertex_collection_names_list = parameters.get('to_vertex_collection_names_list')

                    graph.create_edge_definition(
                        edge_collection=edge_collection_name,
                        from_vertex_collections=from_vertex_collection_names_list,
                        to_vertex_collections=to_vertex_collection_names_list
                    )
                else:
                    print(
                        "ArangoDBDatabaseManager.create_edge_definition - edge_definition: " + edge_collection_name + " already exist in graph: " + graph_name)

        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.create_edge_definition - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def delete_edge_definition(self, graph_name, edge_collection_name, parameters=None):
        try:
            self.test_database_connection()

            if self.has_graph(graph_name):
                graph = self.get_graph(graph_name)
                graph.delete_edge_definition(edge_collection_name, purge=True)
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.create_edge_definition - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def replace_edge_definition(self, graph_name, parameters=None):
        try:
            self.test_database_connection()

            if self.has_graph(graph_name):
                graph = self.get_graph(graph_name)
                edge_collection_name = parameters.get('edge_collection_name')

                if not graph.has_edge_definition(edge_collection_name):
                    from_vertex_collection_names_list = parameters.get('from_vertex_collection_names_list')
                    to_vertex_collection_names_list = parameters.get('to_vertex_collection_names_list')

                    graph.replace_edge_definition(
                        edge_collection=edge_collection_name,
                        from_vertex_collections=from_vertex_collection_names_list,
                        to_vertex_collections=to_vertex_collection_names_list
                    )
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.replace_edge_definition - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def graph_dict_to_db(self, graph_dict, collection_mode='update'):

        is_vertex_of_edges = []
        is_edge_of_edges = []
        db_vertices = {}
        db_edges = {}
        db_edge_definitions = {}

        self.parse_sub_graph(graph_dict, db_vertices, db_edges, db_edge_definitions, is_vertex_of_edges,
                             is_edge_of_edges, parent_type=None, parent_key=None, is_root=True)

        for collection_name in db_vertices.keys():
            db_vertices_list = []
            for vertex_id in db_vertices[collection_name]:
                db_vertices_list.append(db_vertices[collection_name][vertex_id])

            self.integrate_db_collection(collection_name, db_vertices_list, collection_mode)
        for db_edge_collection_name in db_edges.keys():
            db_edges_list = []
            for edge_id in db_edges[db_edge_collection_name]:
                db_edges_list.append(db_edges[db_edge_collection_name][edge_id])

            self.integrate_db_collection(db_edge_collection_name, db_edges_list, collection_mode,
                                         is_edge_collection=True)

        self.integrate_db_collection('is_vertex_of_edges', is_vertex_of_edges, collection_mode, is_edge_collection=True)
        self.integrate_db_collection('is_edge_of_edges', is_edge_of_edges, collection_mode, is_edge_collection=True)

    ##########
    # collection functions
    ##########

    def has_collection(self, collection_name, parameters=None):
        self.test_database_connection()
        return self.database.has_collection(collection_name)

    def get_collection(self, collection_name, parameters=None):
        self.test_database_connection()
        return self.database[collection_name]

    def get_collections(self):
        self.test_database_connection()
        collections = []
        for collection in self.database.collections():
            if not collection['system']:
                collections.append(collection)
        return collections

    def get_edge_collections(self):
        edge_collections = []
        collections = self.get_collections()
        for collection in collections:
            if collection['type'] == 'edge':
                edge_collections.append(collection)
        return edge_collections

    def get_collection_as_a_list(self, collection_name):
        self.test_database_connection()
        documents_list = []
        for item in self.get_collection(collection_name):
            documents_list.append(item)
        return documents_list

    def get_collection_as_a_dict(self, collection_name):
        self.test_database_connection()
        documents_dict = {}
        for item in self.get_collection(collection_name):
            documents_dict[item['_key']] = item
        return documents_dict

    def create_collection(self, collection_name, parameters=None):
        self.test_database_connection()
        print("ArangoDBDatabaseManager.create_collection: " + collection_name)

        edge = parameters.get('is_edge_collection', False) if parameters else False

        return self.database.create_collection(collection_name, edge=edge)

    def delete_collection(self, collection_name, parameters=None):
        try:
            self.test_database_connection()
            if self.database.has_collection(collection_name):
                self.database.delete_collection(collection_name, ignore_missing=True)
                print("ArangoDBDatabaseManager.delete_create_collection: " + collection_name)
            else:
                print("ArangoDBDatabaseManager.delete_collection - collection:: " + collection_name + " does not exist")

        except Exception as e1:
            print("ArangoDBDatabaseManager.delete_collection - raised Exception_1: " + str(e1))
            try:
                self.database.delete_collection(collection_name, False)
            except Exception as e2:
                print("ArangoDBDatabaseManager.delete_collection - raised Exception_2: " + str(e2))

    def truncate_collection(self, collection_name, parameters=None):
        try:
            self.test_database_connection()
            if self.database.has_collection(collection_name):
                self.database[collection_name].truncate()
                print("ArangoDBDatabaseManager.truncate_collection: " + collection_name)
            else:
                print(
                    "ArangoDBDatabaseManager.truncate_collection - collection:: " + collection_name + " does not exist")

        except Exception as e1:
            print("ArangoDBDatabaseManager.truncate_collection - raised Exception: " + str(e1))

    ##########
    # document functions
    ##########

    def has_document(self, collection_name, key, parameters=None):
        self.test_database_connection()
        if self.has_collection(collection_name):
            return self.database[collection_name].has(key)
        else:
            return False

    def has_many_documents(self, collection_name, documents, parameters=None):
        return [doc for doc in documents if self.has_document(collection_name, doc['_key'])]

    def get_document(self, collection_name, key, parameters=None):
        self.test_database_connection()
        return self.database[collection_name][key]

    def replace_document(self, collection_name, document, parameters=None):
        self.test_database_connection()
        self.database[collection_name].replace(document)

    def delete_document(self, collection_name, document, parameters=None):
        self.test_database_connection()
        if parameters is not None:
            self.database[collection_name].delete(document, **parameters)
        else:
            self.database[collection_name].delete(document)

    def delete_many_documents(self, collection_name, documents, parameters=None):
        self.test_database_connection()
        if parameters is not None:
            self.database[collection_name].delete_many(documents, **parameters)
        else:
            self.database[collection_name].delete_many(documents)

    def replace_many_documents(self, collection_name, documents, parameters=None):
        try:
            self.database[collection_name].replace_many(documents,
                                                        check_rev=True,
                                                        return_new=False,
                                                        return_old=False,
                                                        sync=None,
                                                        silent=False)
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.replace_many_documents - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def update_document(self, collection_name, document, parameters=None):
        self.test_database_connection()
        self.database[collection_name].update(document)

    def update_many_documents(self, collection_name, documents, parameters=None):
        self.test_database_connection()
        self.database[collection_name].update_many(documents)

    def insert_document(self, collection_name, document, parameters=None):
        self.test_database_connection()
        try:
            params = {'return_new': False,
                      'sync': None,
                      'silent': False,
                      'overwrite': False,
                      'return_old': False}

            if parameters is not None:
                params.update(parameters)

            self.database[collection_name].insert(document, **params)
        except Exception as e:
            print(
                "ArangoDBDatabaseManager.insert_document raised an Exception: " + str(e) + " doc['_key']: " + document[
                    '_key'])

    def insert_many_documents(self, collection_name, documents, parameters=None):
        try:
            print("ArangoDBDatabaseManager.insert_many_documents - started...", collection_name)
            self.test_database_connection()
            if not self.has_collection(collection_name):
                self.create_collection(collection_name, parameters)

            params = {'return_new': False,
                      'sync': None,
                      'silent': False,
                      'overwrite': False,
                      'return_old': False}

            if parameters is not None:
                params.update(parameters)

            self.database[collection_name].insert_many(documents, **params)
            print("ArangoDBDatabaseManager.insert_many_documents - end")
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("ArangoDBDatabaseManager.insert_many_documents - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    ##########
    # query functions
    ##########

    def get_query(self, query_name):
        self.test_database_connection()
        try:
            return self.database_queries[query_name][self.query_language]
        except Exception as e:
            print("ArangoDBDatabaseManager.get_query Exception: " + str(e))

    def get_query_result(self, query, parameters=None):
        self.test_database_connection()
        aql_wrapper = self.database.aql

        bind_vars = self.get_bind_vars(parameters)

        return aql_wrapper.execute(query, batch_size=100000, ttl=3600, bind_vars=bind_vars)

    def get_bind_vars(self, parameters=None):
        if parameters:
            bind_vars = {}

            for key in parameters.keys():
                if key != 'is_edge_collection':
                    prefix = ''
                    type = parameters[key]['type']
                    if type != 'substitution':
                        if type == 'collection':
                            prefix = '@'
                        bind_vars[prefix + key] = parameters[key]['value']
            return bind_vars
        else:
            return {}

    def execute_query(self, query_name, parameters=None, filters=None):
        if self.is_in_memory:
            cursor = self.database.get_cursor(query_name)
        else:
            query = self.get_query(query_name)
            if filters:
                query = query.replace('return', self.translate_filters(filters) + ' return ')
            if parameters:
                for key in parameters.keys():
                    type = parameters[key]['type']
                    if type == 'substitution':
                        query = query.replace('@' + key, parameters[key]['value'], )

            cursor = self.get_query_result(query, parameters)
        return cursor

    def translate_to_querying_language(self, input):
        return self.querying_language_dictionary.get(input)

    def translate_filters(self, filters):
        query_injection = 'filter '
        filter_count = len(filters)
        filter_counter = 0
        for filter in filters:
            filter_counter += 1
            query_injection += 'document.' + filter['attribute'] + self.translate_to_querying_language(
                filter['operator']) + filter['value']
            if filter_counter < filter_count:
                query_injection += " " + self.translate_to_querying_language(filter["logical_operator"]) + " "  # " && "

        return query_injection

    def get_dictionary_from_cursor(self, cursor, key_attribute='_key'):
        dictionary = {}
        for item in cursor:
            dictionary.update({item[key_attribute]: item})
        return dictionary

    def tuple_to_id(self, tuple):
        key = ''
        for a in tuple:
            key = key + str(a) + ':'
        key = key.replace(' ', '_')
        return key[:-1]

    def traverse(self, graph_name, start_vertex=None,
                 direction='any',
                 min_depth=1,
                 max_depth=2):
        graph = self.database.graph(graph_name)
        result = graph.traverse(
            start_vertex=start_vertex,
            direction=direction,
            min_depth=min_depth,
            max_depth=max_depth)
        return result

    def upsert_edge(self, graph_name, edge_doc):
        self.test_database_connection()
        try:
            edge_key = edge_doc["_key"]
            if self.has_document(graph_name, edge_key):
                if 'weight' in edge_doc.keys():
                    edge_doc['weight'] = self.get_document(graph_name, edge_key)['weight'] + 1
                self.replace_document(graph_name, edge_doc)
            else:
                self.insert_document(graph_name, edge_doc)
        except Exception as e:
            print("Exception caught for insert_document edge_key: " + edge_key + " - exception: " + str(e))


if __name__ == "__main__":
    print('DatabaseManager start')
    filters = [{'attribute': 'core_level', 'operator': '>', 'value': '1', 'logical_operator': 'and'},
               {'attribute': 'is_domain_knowledge', 'operator': '=', 'value': 'true'}]

    query_injection = 'filter '
    filter_count = len(filters)
    filter_counter = 0
    for filter in filters:
        filter_counter += 1
        query_injection += 'document.' + filter['attribute'] + filter['operator'] + filter['value']
        if filter_counter < filter_count:
            query_injection += " " + filter.get("logical_operator") + " "

    print(query_injection)
    print('DatabaseManager end')
