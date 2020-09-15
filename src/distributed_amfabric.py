#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.amfabric import AMFabric
from typing import Optional, Dict, Union, List
from dask.distributed import Client, Variable, Lock,  TimeoutError
from src.sdr import SDR
from src.neuro_column import NeuroColumn
from src.distributed_cache import DistributedCache


class DistributedAMFabric:
    def __init__(self, config: Optional[Dict[str, str]], dask_client: Optional[Client] = None):
        """
        class that implements a distributed version of AMFabric.

        :param config: dictionary containing the following keys:
                'db_name': the database name
                'db_username': database user name
                'db_password': database password
                'db_system': type of database
                'db_config_file_path': path to the database configurations
                'db_queries_file_path': path to the queries configuration
                'fabric_graph': the name of the fabric graph
                'scheduler_address': address of the Dask scheduler
        :param dask_client: a Dask Client - if none a new one will be created
        """
        self.config = config
        if dask_client is None:
            self.client = Client(config['scheduler_address'])
        else:
            self.client = dask_client

        self.cache = DistributedCache(config=config, dask_client=dask_client)

    @staticmethod
    def lock_fabric(fabric_uid: str) -> Lock:
        """
        method to get a lock for the specified area of fabric
        :param fabric_uid: the fabric area
        :return: a distributed lock
        """
        distributed_fabric_uid = 'AMFabric_{}'.format(fabric_uid)
        return Lock(distributed_fabric_uid)

    def create_distributed_fabric(self,
                                  fabric_uid: str,
                                  short_term_memory: int = 1,
                                  mp_threshold: float = 0.1,
                                  structure: str = 'star',
                                  prune_threshold: float = 0.001,
                                  min_cluster_size: int = 2,
                                  cluster_start_threshold: float = 0.5,
                                  cluster_step: float = 0.01,
                                  random_seed: Optional[Union[str, int]] = None,
                                  restore: bool = True) -> bool:
        """
        method to create / restore a distributed AMFabric
        :param fabric_uid: the name of the fabric area
        :param short_term_memory: the length of the short term memory - should be a number greater than 0 - too large will significantly slow down learning and utilize a lot of memory
        :param mp_threshold: this is a multiplier of short_term_memory length and defines the window length from with motifs and anomalies are detected
        :param structure: should be either 'star' - a single neuro_column surrounded by 4 neuro_columns or 'square' - a single neuro_column surrounded by 8 neuro_columns
        :param prune_threshold: a number below which the learning machine will assume learned probabilities are zero
        :param min_cluster_size: minimum number of neuro_columns allowed in a cluster
        :param cluster_start_threshold: starting similarity threshold
        :param cluster_step: increment in similarity threshold during search
        :param random_seed: a seed for the randomisation of the fabric
        :param restore: If True then the fabric will be restored from the database
        :return: True if successfully restored else false
        """

        # lock the fabric area
        #
        fabric_lock = self.lock_fabric(fabric_uid)
        fabric_lock.acquire()

        # create a distributed variable with the AMFabric object
        #
        distributed_fabric_uid = 'AMFabric_{}'.format(fabric_uid)

        # flag indicating if restored from DB
        #
        result = False

        try:
            # try to get the distributed variable containing the distributed AMFabric
            #
            Variable(distributed_fabric_uid).get(timeout=0.1)

        except TimeoutError:

            # the distributed variable does not exist so create it
            #
            fabric_future = self.client.submit(AMFabric,
                                               fabric_uid,
                                               short_term_memory,
                                               mp_threshold,
                                               structure,
                                               prune_threshold,
                                               min_cluster_size,
                                               cluster_start_threshold,
                                               cluster_step,
                                               random_seed,
                                               actor=True)

            # create the distributed variable and store the new fabric actor
            #
            Variable(distributed_fabric_uid).set(fabric_future)

            # only restore if not in cache and asked to
            #
            if restore:

                # get the persistence graph from the distributed cache
                #
                self.cache.create_distributed_store(store_name=self.config['fabric_graph'], key=distributed_fabric_uid, restore=True)
                pg_future = self.cache.get_kv(store_name=self.config['fabric_graph'], key=distributed_fabric_uid)
                pg = pg_future.result()

                # pg will be none if it doesn't exist on database
                #
                if pg is not None:
                    fabric = fabric_future.result()
                    fabric.set_persist_graph(pg=pg)
                    result = True

        fabric_lock.release()

        return result

    def search_for_bmu(self, fabric_uid: str, sdr: SDR, ref_id: Union[str, int], non_hebbian_edges=('generalise',), fast_search: bool = True, lock_fabric: bool = True) -> Optional[dict]:
        """
        method searches for the 'best matching unit' (BMU) of neuro_columns most similar to the last n (short term memory) SDRs presented
        :param fabric_uid: the fabric area to search
        :param sdr: the latest sdr to include in the short term memory
        :param ref_id: the reference id associated with this search
        :param non_hebbian_edges: a tuple of edge_types that are not to be included in the search (ie they are 'non-hebbian')
        :param fast_search: If true a fast approximation for finding the BMU is used else if false a brute force search is used
        :param lock_fabric: If True the fabric will be locked before accessing.If False it assumes a lock has already been acquired
        :return: dictionary of search results
        """

        # create a distributed variable with the AMFabric object
        #
        distributed_fabric_uid = 'AMFabric_{}'.format(fabric_uid)

        try:
            # try to get the distributed variable containing the distributed AMFabric
            #
            dist_fabric_var = Variable(distributed_fabric_uid).get(timeout=0.1)
            fabric = dist_fabric_var.result()

            if lock_fabric:
                fabric_lock = self.lock_fabric(fabric_uid)
                fabric_lock.acquire()
            else:
                fabric_lock = None

            search_por = fabric.search_for_bmu(sdr=sdr, ref_id=ref_id, non_hebbian_edges=non_hebbian_edges, fast_search=fast_search)

            if fabric_lock is not None:
                fabric_lock.release()

        except TimeoutError:
            search_por = None

        return search_por

    def learn(self, fabric_uid: str, search_por: dict, similarity_learn: bool = True, persist: bool = True, lock_fabric: bool = True) -> Optional[dict]:
        """
        Method that learns the short term memory of n last SDRs. Method uses the search results found in search_por to determine which neuro_columns to adapt

        :param fabric_uid: the fabric area that will learn the short term memory
        :param search_por: the search results path of reasoning dictionary
        :param similarity_learn: if true then learning rate is proportional to the similarity,
                            else if false learning rate is proportional to distance (1- similarity)
        :param persist: If true the updated fabric will be persisted
        :param lock_fabric: If True the fabric will be locked before accessing. If False it assumes a lock has already been acquired
        :return: dictionary of search results
        """

        # create a distributed variable with the AMFabric object
        #
        distributed_fabric_uid = 'AMFabric_{}'.format(fabric_uid)

        try:
            # try to get the distributed variable containing the distributed AMFabric
            #
            dist_fabric_var = Variable(distributed_fabric_uid).get(timeout=0.1)
            fabric = dist_fabric_var.result()

            if lock_fabric:
                fabric_lock = self.lock_fabric(fabric_uid)
                fabric_lock.acquire()
            else:
                fabric_lock = None

            # learn the current short_term_memory
            #
            learn_por = fabric.learn(search_por=search_por, similarity_learn=similarity_learn)

            if persist:
                fabric.update_persist_graph(only_updated=True, ref_id=search_por['ref_id'])

            if fabric_lock is not None:
                fabric_lock.release()

        except TimeoutError:
            learn_por = None

        return learn_por

    def train(self, fabric_uid: str, sdr: SDR, ref_id: Union[str, int], non_hebbian_edges=('generalise',), fast_search: bool = True, similarity_learn: bool = True, persist: bool = True,
              lock_fabric: bool = True) -> Optional[dict]:
        """
        method to train the fabric. The provided SDR is added to short term memory of the last n SDRs. The most similar 'best matching unit', BMU, is then found and adapted using a hebbian update rule
        :param fabric_uid: the area of fabric to learn the sequence of SDRs
        :param sdr: the latest SDR to add to the short term memory
        :param ref_id: the reference id associated with the SDR
        :param non_hebbian_edges: the tuple of edge_types that are to be ignored when searching for the BMU
        :param fast_search: If true a fast approximation for finding the BMU is used else if false a brute force search is used
        :param similarity_learn: if true then learning rate is proportional to the similarity,
                            else if false learning rate is proportional to distance (1- similarity)
        :param persist: If True the updated fabric will persisted
        :param lock_fabric: If true the fabric will be locked else if false it is assumed a fabric lock has already been acquired
        :return: Path Of Reasoning POR dictionary
        """

        # create a distributed variable with the AMFabric object
        #
        distributed_fabric_uid = 'AMFabric_{}'.format(fabric_uid)

        try:
            # try to get the distributed variable containing the distributed AMFabric
            #
            dist_fabric_var = Variable(distributed_fabric_uid).get(timeout=0.1)
            fabric = dist_fabric_var.result()

            if lock_fabric:
                fabric_lock = self.lock_fabric(fabric_uid)
                fabric_lock.acquire()
            else:
                fabric_lock = None

            # learn the current short_term_memory
            #
            por = fabric.train(sdr=sdr, ref_id=ref_id, non_hebbian_edges=non_hebbian_edges, fast_search=fast_search, similarity_learn=similarity_learn)
            if persist:
                pg = fabric.update_persist_graph(only_updated=True, ref_id=ref_id)
                pg = pg.result()
            if fabric_lock is not None:
                fabric_lock.release()

            pass
        except TimeoutError:
            por = None

        return por

    def persist_fabric(self, fabric_uid: str, lock_fabric: bool = True):
        """
        method to persist the fabric to the database

        :param fabric_uid: the fabric area to persist
        :param lock_fabric:
        :return:
        """
        # create a distributed variable with the AMFabric object
        #
        distributed_fabric_uid = 'AMFabric_{}'.format(fabric_uid)

        try:
            # try to get the distributed variable containing the distributed AMFabric
            #
            dist_fabric_var = Variable(distributed_fabric_uid).get(timeout=0.1)
            fabric = dist_fabric_var.result()

            if lock_fabric:
                fabric_lock = self.lock_fabric(fabric_uid)
                fabric_lock.acquire()
            else:
                fabric_lock = None

            # persist the fabric if required
            # first get current persist graph from fabric
            # then persist back into cache
            #
            fabric_key = 'AMFabric_{}'.format(fabric_uid)
            pg_future = fabric.get_persist_graph('persist')
            pg = pg_future.result()
            result = self.cache.set_kv(store_name=self.config['fabric_graph'], key=fabric_key, value=pg, persist=True)

            if fabric_lock is not None:
                fabric_lock.release()

        except TimeoutError:
            result = None

        return result

    def query(self, fabric_uid: str, sdr, top_n: int = 1, fast_search: bool = False, lock_fabric: bool = True) -> Optional[Dict[str, Union[List[str], NeuroColumn]]]:
        """
        method to query an area of the fabric.
        :param fabric_uid: the fabric area to query
        :param sdr: a list of SDRs or a single SDR that will be used to earch for the 'best matching unit' BMU neuro_column
        :param top_n: The number of BMus that will be merged (weighted by similarity to the query SDRs) and returned.
        :param fast_search: if true a fast approximation algorithm is used to find the BMU else if False then Brute Force is used
        :param lock_fabric: If true the fabric will be locked else if false it is assumed a fabric lock has already been acquired
        :return: Query results
        """

        # create a distributed variable with the AMFabric object
        #
        distributed_fabric_uid = 'AMFabric_{}'.format(fabric_uid)

        try:
            # try to get the distributed variable containing the distributed AMFabric
            #
            dist_fabric_var = Variable(distributed_fabric_uid).get(timeout=0.1)
            fabric = dist_fabric_var.result()

            if lock_fabric:
                fabric_lock = self.lock_fabric(fabric_uid)
                fabric_lock.acquire()
            else:
                fabric_lock = None

            result = fabric.query(sdr=sdr, top_n=top_n, fast_search=fast_search)

            if fabric_lock is not None:
                fabric_lock.release()

        except TimeoutError:
            result = None

        return result

    def decode_fabric(self, fabric_uid: str, all_details: bool = False, min_cluster_size: int = 1, start_threshold: float = 0.6, step: float = 0.005, lock_fabric: bool = True) -> Optional[dict]:
        """
        method returns a dictionary representation of the fabric neuro_columns and associated statistics
        :param fabric_uid: the area to retrieve
        :param all_details: If true all stats are included else if False just the definitions of the neuro_columns
        :param min_cluster_size: minimum number of neuro_columns allowed in a cluster
        :param start_threshold: starting similarity threshold
        :param step: increment in similarity threshold during search
        :param lock_fabric: If True the fabric will be locked else if false it is assumed a fabric lock has already been acquired
        :return: a dictionary representing the fabric
        """

        # create a distributed variable with the AMFabric object
        #
        distributed_fabric_uid = 'AMFabric_{}'.format(fabric_uid)

        try:
            # try to get the distributed variable containing the distributed AMFabric
            #
            dist_fabric_var = Variable(distributed_fabric_uid).get(timeout=0.1)
            fabric = dist_fabric_var.result()

            if lock_fabric:
                fabric_lock = self.lock_fabric(fabric_uid)
                fabric_lock.acquire()
            else:
                fabric_lock = None

            result = fabric.decode_fabric(all_details=all_details,
                                          min_cluster_size=min_cluster_size,
                                          start_threshold=start_threshold,
                                          step=step)

            if fabric_lock is not None:
                fabric_lock.release()

        except TimeoutError:
            result = None

        return result

    def get_anomalies(self, fabric_uid: str, lock_fabric: bool = True) -> Optional[dict]:
        """
        method to return the anomalies detected since the beginning
        :param fabric_uid: the fabric area to retrieve from
        :param lock_fabric: If True the fabric will be locked else if false it is assumed a fabric lock has already been acquired
        :return: dictionary of anomalies detected
        """

        # create a distributed variable with the AMFabric object
        #
        distributed_fabric_uid = 'AMFabric_{}'.format(fabric_uid)

        try:
            # try to get the distributed variable containing the distributed AMFabric
            #
            dist_fabric_var = Variable(distributed_fabric_uid).get(timeout=0.1)
            fabric = dist_fabric_var.result()

            if lock_fabric:
                fabric_lock = self.lock_fabric(fabric_uid)
                fabric_lock.acquire()
            else:
                fabric_lock = None

            result = fabric.get_anomalies()

            if fabric_lock is not None:
                fabric_lock.release()

        except TimeoutError:
            result = None

        return result

    def get_motifs(self, fabric_uid: str, lock_fabric: bool = True) -> Optional[dict]:
        """
        method to return the motifs detected since the beginning
        :param fabric_uid: the fabric area to retrieve from
        :param lock_fabric: If True the fabric will be locked else if false it is assumed a fabric lock has already been acquired
        :return: dictionary of motifs detected
        """

        # create a distributed variable with the AMFabric object
        #
        distributed_fabric_uid = 'AMFabric_{}'.format(fabric_uid)

        try:
            # try to get the distributed variable containing the distributed AMFabric
            #
            dist_fabric_var = Variable(distributed_fabric_uid).get(timeout=0.1)
            fabric = dist_fabric_var.result()

            if lock_fabric:
                fabric_lock = self.lock_fabric(fabric_uid)
                fabric_lock.acquire()
            else:
                fabric_lock = None

            result = fabric.get_motifs()

            if fabric_lock is not None:
                fabric_lock.release()

        except TimeoutError:
            result = None

        return result

