#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.amfabric import AMFabric
from typing import Optional, Dict, Union, List
from dask.distributed import Client, Variable, Lock, ActorFuture,  TimeoutError
from src.sdr import SDR
from src.neuro_column import NeuroColumn
from src.distributed_cache import DistributedCache


class DistributedAMFabric:
    def __init__(self, config: Optional[Dict[str, Union[str, bool]]], dask_client=None):
        self.config = config
        if dask_client is None:
            self.client = Client(config['scheduler_address'])
        else:
            self.client = dask_client

        self.cache = DistributedCache(config=config, dask_client=dask_client)

    @staticmethod
    def lock_fabric(fabric_name: str) -> Lock:
        distributed_fabric_name = 'AMFabric:{}'.format(fabric_name)
        return Lock(distributed_fabric_name)

    def get_dist_fabric(self, fabric_name: str, restore: bool = False) -> ActorFuture:
        """
        method gets a distributed amfabric from the Dask Scheduler if it exists.
        If it doesnt exists then it creates it first
        :param fabric_name: the fabric name for the cache
        :param restore: If True cache is restored from DB if it doesnt already exist
        :return: an actor future pointing to the distributed AMFabric
        """

        with self.lock_fabric(fabric_name):

            # create a distributed variable with the AMFabric object
            #
            distributed_fabric_name = 'AMFabric:{}'.format(fabric_name)

            # if the distributed variable
            try:
                Variable(distributed_fabric_name).get(timeout=0.1)
            except TimeoutError:
                new_cache_future = self.client.submit(AMFabric,
                                                      self.config['uid'],
                                                      self.config['short_term_memory'],
                                                      self.config['mp_threshold'],
                                                      self.config['structure'],
                                                      self.config['prune_threshold'],
                                                      actor=True)
                Variable(distributed_fabric_name).set(new_cache_future)

            dist_fabric_var = Variable(distributed_fabric_name).get()
            new_fabric_actor = dist_fabric_var.result()

            # restore from DB if it exists
            #
            if restore:
                future = self.cache.get_kv(store_name='HSG', key='AMFabric_{}'.format(self.config['uid']), restore=True)
                pg = future.result()
                if pg is not None:
                    new_fabric_actor.set_persist_graph(pg=pg)

        return new_fabric_actor

    def search_for_bmu(self, sdr: SDR, ref_id: Union[str, int], non_hebbian_edges=('generalise',)) -> dict:

        fabric = self.get_dist_fabric(fabric_name=self.config['uid'])
        search_por = fabric.search_for_bmu(sdr=sdr, ref_id=ref_id, non_hebbian_edges=non_hebbian_edges)

        return search_por

    def learn(self, search_por: dict, persist: bool = True) -> dict:
        fabric = self.get_dist_fabric(fabric_name=self.config['uid'])
        learn_por = fabric.learn(search_por=search_por)

        if persist:
            fabric_key = 'AMFabric_{}'.format(self.config['uid'])
            future = self.cache.get_kv(store_name='HSG', key=fabric_key, restore=True)
            pg = future.result()
            pg = fabric.get_persist_graph(pg_to_update=pg, only_updated=True)
            self.cache.set_kv(store_name='HSG', key=fabric_key, value=pg, persist=True)
        return learn_por

    def train(self, sdr: SDR, ref_id: Union[str, int], non_hebbian_edges=('generalise',), persist: bool = True) -> dict:
        fabric = self.get_dist_fabric(fabric_name=self.config['uid'])
        por = fabric.train(sdr=sdr, ref_id=ref_id, non_hebbian_edges=non_hebbian_edges)
        if persist:
            fabric_key = 'AMFabric_{}'.format(self.config['uid'])
            future = self.cache.get_kv(store_name='HSG', key=fabric_key, restore=True)
            pg = future.result()
            pg = fabric.get_persist_graph(pg_to_update=pg, only_updated=True)
            self.cache.set_kv(store_name='HSG', key=fabric_key, value=pg, persist=True)

        return por

        pass

    def query(self, sdr, top_n: int = 1) -> Dict[str, Union[List[str], NeuroColumn]]:
        fabric = self.get_dist_fabric(fabric_name=self.config['uid'])
        result = fabric.query(sdr=sdr, top_n=top_n)
        return result

    def decode_fabric(self, all_details: bool = False, only_updated: bool=False, community_sdr: bool = False) -> dict:
        fabric = self.get_dist_fabric(fabric_name=self.config['uid'])
        result = fabric.decode_fabric(all_details=all_details, only_updated=only_updated, community_sdr=community_sdr)
        return result

    def get_anomalies(self) -> dict:
        fabric = self.get_dist_fabric(fabric_name=self.config['uid'])
        result = fabric.get_anomalies()
        return result

    def get_motifs(self) -> dict:
        fabric = self.get_dist_fabric(fabric_name=self.config['uid'])
        result = fabric.get_motifs()
        return result

