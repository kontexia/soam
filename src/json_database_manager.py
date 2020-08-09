from src.database_manager import DatabaseManager
import shutil
import json
import os
import os.path


class JsonDatabaseManager(DatabaseManager):
    database_URL = None
    database_name = None
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
                                    '!~': '!~'}

    def __init__(self, parameters=None):

        self.os_username = parameters.get('os_username', None)
        self.database_system = parameters.get('database_system', 'json_db')
        self.database_name = parameters.get('database_name', '_system')
        self.database_queries = self.get_database_queries(parameters.get('database_queries_file_path'))
        self.database_connection_parameters = self.get_database_connection_parameters(
            parameters.get('databases_configuration_file_path'),
            self.database_system,
            self.database_name)
        if self.database_connection_parameters:
            self.username = self.database_connection_parameters['username']
            self.database_URL = self.database_connection_parameters['database_URL'].replace('username',
                                                                                            self.os_username)
            self.password = self.database_connection_parameters['password']
            self.query_language = self.database_connection_parameters['query_language']

    ##########
    # database functions
    ##########
    def get_database_connection_parameters(self,
                                           databases_configuration_file_path=None,
                                           database_system='json',
                                           database_name='_system'):
        try:
            with open(databases_configuration_file_path, 'r') as readfile:
                databases_configuration = json.load(readfile)
            return databases_configuration[database_system][database_name]
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("JsonDatabaseManager.get_database_connection_parameters - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def create_database_connection(self):
        try:
            self.database = InMemoryDatabase()
            self.database.set_db_path(self.database_URL + self.database_name + '/')
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("JsonDatabaseManager.create_database_connection - raised an Exception: " + str(e))
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
            self.database.create_database(database_name)
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
        edge = parameters.get('is_edge_collection', False) if parameters else False
        return self.database.has_collection(collection_name, edge=edge)

    def get_collection(self, collection_name, parameters=None):
        self.test_database_connection()
        edge = parameters.get('is_edge_collection', False) if parameters else False
        return self.database.load_from_disk(collection_name, edge=edge)
        # return self.database[collection_name]

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
                self.database.delete_collection(collection_name)
                print("ArangoDBDatabaseManager.delete_create_collection: " + collection_name)
            else:
                print("ArangoDBDatabaseManager.delete_collection - collection:: " + collection_name + " does not exist")

        except Exception as e1:
            print("ArangoDBDatabaseManager.delete_collection - raised Exception_1: " + str(e1))
            try:
                self.database.delete_collection(collection_name, False)
            except Exception as e2:
                print("ArangoDBDatabaseManager.delete_collection - raised Exception_2: " + str(e2))

    ##########
    # document functions
    ##########

    def has_document(self, collection_name, key, parameters=None):
        self.test_database_connection()
        if self.has_collection(collection_name, parameters):
            collection = self.get_collection(collection_name)
            return key in collection.keys()
            # return self.database[collection_name].has(key)
        else:
            return False

    def has_many_documents(self, collection_name, documents, parameters=None):
        return [doc for doc in documents if self.has_document(collection_name, doc['_key'])]

    def get_document(self, collection_name, key, parameters=None):
        self.test_database_connection()
        collection = self.get_collection(collection_name, parameters)
        return collection[key]

    def replace_document(self, collection_name, document, parameters=None):
        self.test_database_connection()
        collection = self.get_collection(collection_name, parameters)
        collection[document['_key']] = document
        edge = parameters.get('is_edge_collection', False) if parameters else False
        self.database.save_on_disk(collection_name, collection, edge=edge)

    def replace_many_documents(self, collection_name, documents, parameters=None):
        try:
            print("JsonDatabaseManager.replace_many_documents - started...")
            self.test_database_connection()
            if not self.has_collection(collection_name, parameters):
                self.create_collection(collection_name, parameters)
            collection = self.get_collection(collection_name)
            for document in documents:
                collection[document['_key']] = document
            edge = parameters.get('is_edge_collection', False) if parameters else False

            self.database.save_on_disk(collection_name, collection, edge=edge)

            '''self.database[collection_name].replace_many(documents,
                                                        check_rev=True,
                                                        return_new=False,
                                                        return_old=False,
                                                        sync=None,
                                                        silent=False)'''
            print("JsonDatabaseManager.insert_many_documents - end")
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
            collection = self.get_collection(collection_name)
            collection[document['_key']] = document
            edge = parameters.get('is_edge_collection', False) if parameters else False
            self.database.save_on_disk(collection_name, collection, edge=edge)
            '''self.database[collection_name].insert(document,
                                                  return_new=False,
                                                  sync=None,
                                                  silent=False,
                                                  overwrite=False,
                                                  return_old=False)'''
        except Exception as e:
            print("JsonDatabaseManager.insert_document raised an Exception: " + str(e) + " doc['_key']: " + document[
                '_key'])

    def insert_many_documents(self, collection_name, documents, parameters=None):
        try:
            print("JsonDatabaseManager.insert_many_documents - started...")
            self.test_database_connection()
            if not self.has_collection(collection_name, parameters):
                self.create_collection(collection_name, parameters)
            collection = self.get_collection(collection_name, parameters)
            for document in documents:
                collection[document['_key']] = document
            edge = parameters.get('is_edge_collection', False) if parameters else False

            self.database.save_on_disk(collection_name, collection, edge=edge)
            '''self.database[collection_name].insert_many(documents,
                                                       return_new=False,
                                                       sync=None,
                                                       silent=False,
                                                       overwrite=False,
                                                       return_old=False)'''
            print("JsonDatabaseManager.insert_many_documents - end")
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("JsonDatabaseManager.insert_many_documents - raised an Exception: " + str(e))
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

        return aql_wrapper.execute(query, bind_vars=bind_vars)

    def get_bind_vars(self, parameters=None):
        if parameters:
            bind_vars = {}

            for key in parameters.keys():
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
        if True:
            self.test_database_connection()
            cursor = self.database.get_cursor(query_name, parameters)
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


class InMemoryDatabase(dict):
    database_name = None
    db_path = None

    def save_on_disk(self, collection_name, collection, edge=False):
        try:
            with open(self.db_path + ('edge/' if edge else 'vertex/') + collection_name + '.json', 'w') as fp:
                json.dump(collection, fp)
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("InMemoryDatabase.save_on_disk - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def load_from_disk(self, collection_name, edge=False):
        try:
            with open(self.db_path + ('edge/' if edge else 'vertex/') + collection_name + '.json', 'r') as fp:
                return json.load(fp)
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("InMemoryDatabase.load_from_disk - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def delete_from_disk(self, collection_name, edge=False):
        try:
            if os.path.isfile(self.db_path + ('edge/' if edge else 'vertex/') + collection_name + '.json'):
                os.remove(self.db_path + ('edge/' if edge else 'vertex/') + collection_name + '.json')
            else:
                print("InMemoryDatabase.delete_from_disk - collection not found:",
                      self.db_path + ('edge/' if edge else 'vertex/') + collection_name + '.json')
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("InMemoryDatabase.delete_from_disk - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    ##########
    # database functions
    ##########
    def set_db_path(self, db_path):
        self.db_path = db_path

    def delete_database(self, database_name):
        try:
            if os.path.isdir(database_name):
                shutil.rmtree(self.db_path + database_name)
            else:
                print("InMemoryDatabase.delete_database - could not find database: " + str(database_name))

        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("InMemoryDatabase.delete_database - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def create_database(self, database_name, users=None):
        try:
            if os.path.isdir(database_name):
                print("InMemoryDatabase.delete_database - database: ", str(database_name), "already exists!")
            else:
                os.makedirs(self.db_path + database_name)
                os.makedirs(self.db_path + database_name + '/vertex')
                os.makedirs(self.db_path + database_name + '/edge')
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("InMemoryDatabase.create_database - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    ##########
    # collection functions
    ##########

    def collections(self):
        return self.values()

    def create_collection(self, collection_name, edge=False):
        self[collection_name] = {}
        self.save_on_disk(collection_name, self[collection_name], edge)

    def delete_collection(self, collection_name):
        self.delete_from_disk(collection_name)

    def has_collection(self, collection_name, edge=False):
        try:
            return os.path.isfile(self.db_path + ('edge/' if edge else 'vertex/') + collection_name + '.json')
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("InMemoryDatabase.has_collection - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")

    def get_cursor(self, query_name, parameters=None):

        try:
            if query_name == 'get_all_doc_in_special_character_dictionaries_for_language':
                language = parameters.get('value')['value']
                collection = self.load_from_disk('special_character_dictionaries')
                return collection.values()

            if query_name == 'get_all_doc_in_contraction_dictionaries_for_language':
                language = parameters.get('value')['value']
                collection = self.load_from_disk('contraction_dictionaries')
                return collection.values()

            if query_name == 'get_all_doc_in_texting_language_dictionaries_for_language':
                language = parameters.get('value')['value']
                collection = self.load_from_disk('texting_language_dictionaries')
                return collection.values()

            if query_name == 'get_all_doc_in_domain_knowledge_dictionaries_for_language':
                language = parameters.get('value')['value']
                collection = self.load_from_disk('domain_knowledge_dictionaries')
                return collection.values()

            if query_name == 'get_all_doc_in_excluded_stop_word_lists_for_language':
                language = parameters.get('value')['value']
                collection = self.load_from_disk('excluded_stop_word_lists')
                return collection.values()

            if query_name == 'get_excluded_stop_words_for_language':
                language = parameters.get('value')['value']
                collection = self.load_from_disk('excluded_stop_word_lists')
                return collection.values()

            if query_name == 'get_all_documents_from_collection':
                collection_name = parameters.get('collection_name')['value']
                edge = parameters.get('is_edge_collection', False) if parameters else False
                collection = self.load_from_disk(collection_name, edge=edge)
                return collection.values()
            '''
            'retrieve_and_update_last_read_message': {
                'AQL': 'FOR topic in @@message_topics_collection_name FILTER topic._key == @topic_key '
                       'UPDATE topic WITH {"subscribers": {@subscriber_key: {"last_read_message_id": topic.message_count}}} '
                       'IN @@message_topics_collection_name OPTIONS {ignoreRevs: false} return OLD'},'''
            if query_name == 'retrieve_and_update_last_read_message':
                collection_name = parameters.get('message_topics_collection_name')['value']
                topic_key = parameters.get('topic_key')['value']
                subscriber_key = parameters.get('subscriber_key')['value']

                collection = self.load_from_disk(collection_name)
                returned_topic = collection[topic_key].copy()
                collection[topic_key]['subscribers'] = {
                    subscriber_key: {"last_read_message_id": collection[topic_key]['message_count']}}
                self.save_on_disk(collection_name, collection)
                return [returned_topic]
            '''
            'increment_and_retrieve_message_topic': {
                'AQL': 'FOR topic IN @@message_topics_collection_name FILTER topic._key == @topic_key '
                       'UPDATE topic WITH {"message_count": topic.message_count + 1} '
                       'IN @@message_topics_collection_name OPTIONS {ignoreRevs: false} RETURN NEW'}'''

            if query_name == 'increment_and_retrieve_message_topic':
                collection_name = parameters.get('message_topics_collection_name')['value']
                topic_key = parameters.get('topic_key')['value']
                subscriber_key = parameters.get('subscriber_key')['value']

                collection = self.load_from_disk(collection_name)
                collection[topic_key]['subscribers'] = {"message_count": collection[topic_key]['message_count'] + 1}
                self.save_on_disk(collection_name, collection)
                return [collection[topic_key]]
            '''
            'get_latest_messages': {
                'AQL': 'FOR msg IN @@message_queue_collection_name FILTER msg.message_id > @last_message_id AND msg.topic_key == @topic_key return msg'},
            '''
            'get_latest_messages'
            if query_name == 'get_latest_messages':
                collection_name = parameters.get('message_queue_collection_name')['value']
                last_message_id = parameters.get('last_message_id')['value']
                topic_key = parameters.get('topic_key')['value']

                collection = self.load_from_disk(collection_name)
                msg_list = []
                for msg in collection:
                    if msg['message_id'] > last_message_id and msg['topic_key'] == topic_key:
                        msg_list.append(msg)

                return msg_list

            '''{'AQL': 'FOR id in split(@conversation_ids,",") FOR conv IN raw_conversations FILTER conv._id == id Return conv'},'''
            if query_name == 'get_raw_conversation_by_ids':
                conversation_ids = parameters.get('conversation_ids')['value']
                collection = self.load_from_disk('raw_conversations')
                conversations = []
                for conv_key in collection:
                    if 'raw_conversations/' + collection[conv_key]['_key'] in conversation_ids:
                        conversations.append(collection[conv_key])
                return conversations
        except Exception as e:
            print("! ! ! ! ! ! ! ! ! !")
            print("InMemoryDatabase.get_cursor - raised an Exception: " + str(e))
            print("! ! ! ! ! ! ! ! ! !")


if __name__ == "__main__":
    print('UniversalDBManager start')
    collection = {'test': 'test__name__'}
    path = r'C:\Users\fontbrua\PycharmProjects\hsg\data_fabric\json_db\demo_db\collection_name.json'
    with open(path, 'w') as fp:
        json.dump(collection, fp)
    print('UniversalDBManager end')
