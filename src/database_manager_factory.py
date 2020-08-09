from src.arango_db_database_manager import ArangoDBDatabaseManager
from src.json_database_manager import JsonDatabaseManager
import os


class DatabaseManagerFactory:

    @staticmethod
    def get_database_manager(username,
                             database_system='arango_db',
                             database_name='_system',
                             config_file_path=None,
                             queries_file_path=None):

        if config_file_path is not None:
            databases_configuration_file_path = os.path.expanduser(config_file_path)
        else:
            databases_configuration_file_path = '../data_fabric/databases_configuration.json'

        if queries_file_path is not None:
            database_queries_file_path = os.path.expanduser(queries_file_path)
        else:
            database_queries_file_path = '../data_fabric/database_queries.json'

        if database_system == 'arango_db':
            return ArangoDBDatabaseManager(
                parameters={'databases_configuration_file_path': databases_configuration_file_path,
                            'database_system': database_system,
                            'database_name': database_name,
                            'database_queries_file_path': database_queries_file_path})

        elif database_system == 'json':
            return JsonDatabaseManager(
                parameters={'os_username': username,
                            'databases_configuration_file_path': databases_configuration_file_path,
                            'database_system': database_system,
                            'database_name': database_name,
                            'database_queries_file_path': database_queries_file_path})

        elif not database_system:
            return JsonDatabaseManager()

