#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Dict
from dask.distributed import Client, Variable, Lock, ActorFuture, Future, TimeoutError, Pub, Sub
from src.kv_cache import KVGraphCache, KVCacheValueType


class DistributedCache:
    def __init__(self, config: Optional[Dict[str, str]]):
        self.config = config
        self.client = Client(config['scheduler_address'])

    def get_dist_cache(self, store_name: str) -> ActorFuture:
        """
        method gets a distributed cache from the Dasj Scheduler if it exists.
        If it doesnt exists it creates it first
        :param store_name: the store name for the cache
        :return:
        """

        with Lock('{}'.format(store_name)):
            try:
                Variable(store_name).get(timeout=0.1)
            except TimeoutError:
                new_cache_future = self.client.submit(KVGraphCache, self.config, actor=True)
                Variable(store_name).set(new_cache_future)

        dist_store_var = Variable(store_name).get()
        new_cache_actor = dist_store_var.result()
        return new_cache_actor

    def set_kv(self, store_name: str, key: str, value: KVCacheValueType, persist=True) -> Future:
        """
        method to add a key value pair to a store. If key already exists then will overwrite

        :param store_name: the store name
        :param key: the unique key within the store
        :param value: the data to save
        :param persist: If true the changes will be persisted in database
        :return: Future
        """
        cache = self.get_dist_cache(store_name=store_name)
        with Lock('{}'.format(store_name)):
            result = cache.set_kv(store_name=store_name, key=key, value=value)
            if persist:
                cache.persist(store_name=store_name)

        # publish key that has changed
        #
        pub = Pub(store_name)
        pub.put({'action': 'set', 'key': key})

        return result

    def del_kv(self, store_name: str, key: Optional[str] = None, persist=True) -> Future:
        """
        method to delete a key within a store or the whole store. If the store/key has been persisted then the delete will be propagated when persist() is called

        :param store_name: the store name
        :param key: the key to delete
        :param persist: If true the changes will be persisted in database
        :return: None
        """
        cache = self.get_dist_cache(store_name=store_name)
        with Lock('{}'.format(store_name)):
            result = cache.del_kv(store_name=store_name, key=key, delete_from_db=persist)
            if persist:
                cache.persist(store_name=store_name)

        # publish key that has changed
        #
        pub = Pub(store_name)
        pub.put({'action': 'delete', 'key': key})

        return result

    def get_kv(self, store_name: str, key: Optional[str] = None) -> KVCacheValueType:
        """
        method to retrieve a stored value
        :param store_name: the store name
        :param key: the key within the store
        :return: KVCacheValueType - the value stored
        """
        cache = self.get_dist_cache(store_name=store_name)
        with Lock('{}'.format(store_name)):
            result = cache.get_kv(store_name=store_name, key=key)
        return result

    def restore(self, store_name, include_history: bool = False) -> bool:
        """
        method to restore a store_name and optionally specific key
        :param store_name: the store_name to restore
        :param include_history: If True restore all history
        :return: bool
        """
        cache = self.get_dist_cache(store_name=store_name)
        with Lock('{}'.format(store_name)):
            result = cache.restore(store_name=store_name, include_history=include_history)
        return result

    def subscribe(self, store_name, timeout=None):
        """
        method to subscribe to any changes to a particular store. Will wait for messages for that store or return after specified timeout
        :param store_name: the storename to listen to
        :param timeout: the timeout to wait for messages
        :return: the message which will be a dict with keys: {'action': set or delete, 'key': the key that changed}
        """
        sub = Sub(store_name)
        return sub.get(timeout=timeout)
