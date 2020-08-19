#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Dict, Union
from dask.distributed import Client, Variable, Lock, ActorFuture, Future, TimeoutError, Pub, Sub
from src.kv_cache import KVGraphCache, KVCacheValueType


class DistributedCache:
    def __init__(self, config: Optional[Dict[str, Union[str, bool]]], dask_client=None):
        """
        Class that implements a simple distributed key value cache that takes care of persisting and restoring to database
        :param config: keys required:\n
                                db_username: the username to login to db with
                                db_system: the db type - arango-db
                                db_name: the name of the database
                                db_config_file_path: the path with the database configuration
                                db_queries_file_path: the path with the query definitions
                                named_query_collection - postfix for a store's named query collection
                                scheduler_address - the address of the Dask Cluster Scheduler
        """

        self.config = config
        if dask_client is None:
            self.client = Client(config['scheduler_address'])
        else:
            self.client = dask_client

    @staticmethod
    def lock_store(store_name: str) -> Lock:
        """
        method to create a distributed lock on a complete store
        :param store_name: the store to lock
        :return: lock object to use within a with statement
        """
        distributed_store_name = 'KVCache:{}'.format(store_name)
        return Lock(distributed_store_name)

    @staticmethod
    def lock_key(store_name: str, key: str) -> Lock:
        """
        method to create a distributed lock on a key within a store
        :param store_name: the store to lock
        :param key: the key in the store to lock
        :return: lock object to use within a with statement
        """
        distributed_store_name_key = 'KVCache:{}:{}'.format(store_name, key)
        return Lock(distributed_store_name_key)

    def get_dist_cache(self, store_name: str, restore: bool = False) -> ActorFuture:
        """
        method gets a distributed cache from the Dask Scheduler if it exists.
        If it doesnt exists then it creates it first
        :param store_name: the store name for the cache
        :param restore: If True cache is restored from DB if it doesnt already exist
        :return: an actor future pointing to the distributed KVCache
        """

        with self.lock_store(store_name):

            # create a distributed variable with the kv_cache object
            #
            distributed_store_name = 'KVCache:{}'.format(store_name)

            # if the distributed variable
            try:
                Variable(distributed_store_name).get(timeout=0.1)
            except TimeoutError:
                new_cache_future = self.client.submit(KVGraphCache, self.config, actor=True)
                Variable(distributed_store_name).set(new_cache_future)

            dist_store_var = Variable(distributed_store_name).get()
            new_cache_actor = dist_store_var.result()

            # restore from DB if it exists
            #
            if restore:
                new_cache_actor.restore(store_name=store_name, include_history=False)

        return new_cache_actor

    def set_kv(self, store_name: str, key: str, value: KVCacheValueType, persist=True) -> ActorFuture:
        """
        method to add a key value pair to a store. If key already exists then will overwrite

        :param store_name: the store name
        :param key: the unique key within the store
        :param value: the data to save
        :param persist: If true the changes will be persisted in database
        :return: ActorFuture with True if success else False
        """

        # if we wish to update existing the restore first if it doesnt exist
        #
        cache = self.get_dist_cache(store_name=store_name)
        result = cache.set_kv(store_name=store_name, key=key, value=value)
        if persist:
            cache.persist(store_name=store_name)

        # publish key that has changed
        #
        pub = Pub(store_name)
        pub.put({'action': 'set', 'key': key})

        return result

    def del_kv(self, store_name: str, key: Optional[str] = None, persist=True) -> ActorFuture:
        """
        method to delete a key within a store or the whole store. If the store/key has been persisted then the delete will be propagated when persist() is called

        :param store_name: the store name
        :param key: the key to delete
        :param persist: If true the changes will be persisted in database
        :return: ActorFuture with True if success else False
        """
        cache = self.get_dist_cache(store_name=store_name)
        result = cache.del_kv(store_name=store_name, key=key, delete_from_db=persist)
        if persist:
            cache.persist(store_name=store_name)

        # publish key that has changed
        #
        pub = Pub(store_name)
        pub.put({'action': 'delete', 'key': key})

        return result

    def get_kv(self, store_name: str, key: Optional[str] = None, restore: bool = True) -> ActorFuture:
        """
        method to retrieve a stored value
        :param store_name: the store name
        :param key: the key within the store
        :param restore: If true DB is checked if store doesnt already exist
        :return: ActorFuture with either KVCacheValueType or None if it doesnt exist
        """
        cache = self.get_dist_cache(store_name=store_name, restore=restore)
        result = cache.get_kv(store_name=store_name, key=key)
        return result

    def restore(self, store_name, include_history: bool = False) -> bool:
        """
        method to restore a store_name and optionally specific key
        :param store_name: the store_name to restore
        :param include_history: If True restore all history
        :return: ActorFuture with True if restored else False
        """
        cache = self.get_dist_cache(store_name=store_name)
        result = cache.restore(store_name=store_name, include_history=include_history)
        return result

    @staticmethod
    def listen_for_updates(store_name, timeout=None):
        """
        method to listen for any changes to a particular store. Will wait for messages for that store or return after specified timeout
        :param store_name: the store name to listen to
        :param timeout: the timeout to wait for messages
        :return: the message which will be a dict with keys: {'action': set or delete, 'key': the key that changed}
        """
        sub = Sub(store_name)
        return sub.get(timeout=timeout)
