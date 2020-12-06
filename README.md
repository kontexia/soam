# SOAM
Python 3.7+ Implementation of Self Organising Associative Memory that can use Dask to scale and Cython for performance optimisations.
SOAM is an example of an explainable Neural Network that uses online Hebbian learning of Network Graphs to cluster, classify and predict.

Library can be used as pure Python or cythonised by running:

        python setup.py build_ext -if 

** NOTE ** You must create the following files in the project top level
* .env  - use the .env_example file as a template to define environment variables with ArangoDB credentials
* db_config.json  - use the db_config_example.json file as a template, replacing ArangoDB database config and credentials as required  
 


## Src Modules

### Database Modules
* database_manager - parent class defining interface for specific database classes
* arango_db_database_manager.py - implementation specific to ArangoDB
* json_db_database_manager.py - implementation specific to json files
* database_manager_factory.py - responsible for instantiating a specific database client given config
* database_queries.json - defines pre-canned queries

### Network Graph Modules
* amgraph.py - implements AMFGraph class (based on Networkx Multi Directional Graph) providing auditing of updates
* sdr.py - implements SDR - a sparse data representation of a network graph suitable for training AMFabric
* subgraph.py - implements SubGraph - a sparse data reprersentation of a netwrok graph suitable for training Neuro-Gas fabric

### In-memory Cache
* kv_cache.py - implements a simple key value cache that provides persistence and understands AMFGraph 
* distributed_kv_cache.py - a Dask distributed key value cache

### Publish & Subscribe
* PubSub - implements a simple Dask publish and subscribe class

### SOAM Modules
* neuro_column.py - Cython optimised implementation of  a column of neurons that forms the basic unit of the AMFabric
* neural_fabric.py - Cython optimised implementation of a fabric of neuro_columns
* amfabric.py - implementation of an online, self organising associative memory fabric
* distributed_amfabric - a Dask distributed AMFabric
* neuro_gas_fabric - implementation of the neuro-gas fabric model (defined as functions to facilitate Dask version)
* distibuted_neuro_gas - a Dask distributed neuro-gas fabric

## Tests Modules
* rainbow_trade_classifier - reads in rainbow trade data from /data dir and trains an in-memory AMFabric
* distributed_amfabric_test - reads in rainbow trade data from /data dir and trains a Dask distributed AMFabric
* neuro_gas_test - tests for neuro-gas fabric
* training_publisher - publisher to train distributed neuro-gas fabric

## Tutorial Jupyter Lab Notebooks
** NOTE ** notebooks rely on Plotly Jupyter Lab extensions that require installation of Node.js and access to npm.

see: https://plotly.com/python/getting-started/#jupyterlab-support-python-35
 

* db_admin.ipynb - low level example of database classes saving nodes and edges to ArangoDB
* publisher.ipynb & subscriber.ipynb - uses Dask to implement simple publish and subscribe
* simulating_data.ipynb - creates test "rainbow trades"
* network_graphs.ipynb - demonstration of AMFGraph class
* amfabric.ipynb - demonstration of the in-memory AMFabric class
* cache.ipynb - demonstration of the in-memory / Dask distributed cache classes and persistence to ArangoDB 
* persisting_amfabric.ipynb - demonstration of persisting the in-memory AMFabric class to ArangoDB
* distributed_amfabric.ipynb - demonstration of the Dask distributed AMFabric class
* soam_viz.ipynb - demonstration of querying the Dask distributed AMFabric class for structural learning

## Data directory
* rainbow_trades.json - simplistic trades with 3 numeric attributes (RGB values), a colour label and trade ID

Comments and issues:

stephen.hancock@kontexia.io

@copyright Kontexia Limited 2020
