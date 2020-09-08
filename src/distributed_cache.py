#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Dict, Union, List
from dask.distributed import Client, Variable, Lock, ActorFuture, TimeoutError
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

    def create_distributed_store(self, store_name: str, key: Optional[Union[List[str], str]] = None, restore: bool = True):
        """
        method to initialise a store and restore
        :param store_name: the store_name
        :param key: a key or list of keys to restore
        :param restore: If true then restore from database
        :return: True if restored else false
        """
        store_lock = self.lock_store(store_name=store_name)
        store_lock.acquire()

        # create a distributed variable with the kv_cache object
        #
        distributed_store_name = 'KVCache_{}'.format(store_name)

        # flag indicating if restored from DB
        #
        result = False

        try:
            # try to get the cache
            #
            Variable(distributed_store_name).get(timeout=0.1)

        except TimeoutError:

            # the distributed variable does not exist so create it
            #
            cache_future = self.client.submit(KVGraphCache, self.config, actor=True)

            # create the distributed variable and store the new cache actor
            #
            Variable(distributed_store_name).set(cache_future)

            # only restore if not in cache and asked to
            #
            if restore:
                cache = cache_future.result()
                cache.restore(store_name=store_name, key=key, include_history=False)
                result = True

        store_lock.release()

        return result

    def set_kv(self, store_name: str, key: str, value: KVCacheValueType, persist: bool = True, lock_cache: bool = True) -> ActorFuture:
        """
        method to add a key value pair to a store. If key already exists then will overwrite

        :param store_name: the store name
        :param key: the unique key within the store
        :param value: the data to save
        :param persist: If true the changes will be persisted in database
        :param lock_cache: If True the cache will be locked before accessing. If False it assumes a lock has already been acquired
        :return: ActorFuture with True if success else False
        """

        distributed_store_name = 'KVCache_{}'.format(store_name)

        try:
            dist_cache_var = Variable(distributed_store_name).get(timeout=0.1)
            cache = dist_cache_var.result()

            if lock_cache:
                cache_lock = self.lock_key(store_name=store_name, key=key)
                cache_lock.acquire()
            else:
                cache_lock = None

            result = cache.set_kv(store_name=store_name, key=key, value=value)
            if persist:
                cache.persist(store_name=store_name)

            if cache_lock is not None:
                cache_lock.release()

        except TimeoutError:
            result = None

        return result

    def del_kv(self, store_name: str, key: Optional[str] = None, persist=True, lock_cache: bool = True) -> ActorFuture:
        """
        method to delete a key within a store or the whole store. If the store/key has been persisted then the delete will be propagated when persist() is called

        :param store_name: the store name
        :param key: the key to delete
        :param persist: If true the changes will be persisted in database
        :param lock_cache: If True the cache will be locked before accessing. If False it assumes a lock has already been acquired
        :return: ActorFuture with True if success else False
        """

        distributed_store_name = 'KVCache_{}'.format(store_name)

        try:
            dist_cache_var = Variable(distributed_store_name).get(timeout=0.1)
            cache = dist_cache_var.result()

            if lock_cache:
                cache_lock = self.lock_key(store_name=store_name, key=key)
                cache_lock.acquire()
            else:
                cache_lock = None

            result = cache.del_kv(store_name=store_name, key=key, delete_from_db=persist)
            if persist:
                cache.persist(store_name=store_name)

            if cache_lock is not None:
                cache_lock.release()

        except TimeoutError:
            result = None

        return result

    def get_kv(self, store_name: str, key: Optional[str] = None, lock_cache: bool = True) -> ActorFuture:
        """
        method to retrieve a stored value
        :param store_name: the store name
        :param key: the key within the store
        :param lock_cache: If True the cache will be locked before accessing. If False it assumes a lock has already been acquired
        :return: ActorFuture with either KVCacheValueType or None if it doesnt exist
        """

        distributed_store_name = 'KVCache_{}'.format(store_name)

        try:
            dist_cache_var = Variable(distributed_store_name).get(timeout=0.1)
            cache = dist_cache_var.result()

            if lock_cache:
                cache_lock = self.lock_key(store_name=store_name, key=key)
                cache_lock.acquire()
            else:
                cache_lock = None

            result = cache.get_kv(store_name=store_name, key=key)

            if cache_lock is not None:
                cache_lock.release()

        except TimeoutError:
            result = None

        return result

    def restore(self, store_name, include_history: bool = False, lock_cache: bool = True) -> bool:
        """
        method to restore a store_name and optionally specific key
        :param store_name: the store_name to restore
        :param include_history: If True restore all history
        :param lock_cache: If True the cache will be locked before accessing. If False it assumes a lock has already been acquired
        :return: ActorFuture with True if restored else False
        """

        distributed_store_name = 'KVCache_{}'.format(store_name)

        try:
            dist_cache_var = Variable(distributed_store_name).get(timeout=0.1)
            cache = dist_cache_var.result()

            if lock_cache:
                cache_lock = self.lock_store(store_name=store_name)
                cache_lock.acquire()
            else:
                cache_lock = None

            result = cache.restore(store_name=store_name, include_history=include_history)

            if cache_lock is not None:
                cache_lock.release()

        except TimeoutError:
            result = None

        return result
