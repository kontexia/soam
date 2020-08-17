#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Dict, Callable, Union
from dask.distributed import Client, Pub, Sub
from src.distributed_cache import DistributedCache
import time


PubSubMsgType = Union[dict, str, float, int, list]
""" type of data that can be published as a message - must be serialisable"""


class PubSub:
    def __init__(self, uid, config: Optional[Dict[str, str]]):
        """
        class that provides an interface to pub sub infrastructure
        :param uid: the unique id of this application
        :param config: keys required:\n
                                db_username: the username to login to db with
                                db_system: the db type - arango-db
                                db_name: the name of the database
                                db_config_file_path: the path with the database configuration
                                db_queries_file_path: the path with the query definitions
                                named_query_collection - postfix for a store's named query collection
                                scheduler_address - the address of the Dask Cluster Scheduler
        """

        self.uid = uid
        """ the uid of the publisher / subscriber """

        self.config = config
        """ the configuration """

        self.client = Client(config['scheduler_address'])
        """ dask client """

        self.cache = DistributedCache(config=config)
        """ a distributed cache to store messages and metadata """

        self.subscriptions = {}
        """ in memory cache of Dask subscription objects keyed by topic"""

        self.publications = {}
        """ in memory cache of Dask publish objects keyed by topic"""

    def close(self) -> None:
        """
        method to close the subscriber correctly - removes this app from the list of current subscribers
        :return: None
        """
        for topic in self.subscriptions.keys():
            with self.cache.lock_key(store_name='subscribers', key=topic):
                subscribers = self.cache.get_kv(store_name='subscribers', key=topic, restore=False).result()
                if subscribers is not None:
                    if self.uid in subscribers:
                        subscribers.remove(self.uid)
                    self.cache.set_kv(store_name='subscribers', key=topic, value=subscribers, persist=False)

    def subscribe(self, topic: str, callback: Callable[[dict], None]) -> None:
        """
        method to subscribe to a topic and register a function to be called when a message arrives

        :param topic: the topic to subscribe to
        :param callback: the callback function to be called when a message arrives
        :return: None
        """
        if topic not in self.subscriptions:
            self.subscriptions[topic] = {'sub': Sub(topic),
                                         'callback': callback}

            # add myself to the list of current subscribers
            #
            with self.cache.lock_key(store_name='subscribers', key=topic):
                subscribers = self.cache.get_kv(store_name='subscribers', key=topic, restore=False).result()
                if subscribers is None:
                    subscribers = [self.uid]
                else:
                    subscribers.append(self.uid)
                self.cache.set_kv(store_name='subscribers', key=topic, value=subscribers, persist=False)

    def unsubscribe(self, topic: str) -> None:
        """
        method to unsubscribe from a topic
        :param topic: the topic to unscribe
        :return: None
        """
        if topic in self.subscriptions:

            # need to remove myself from the list of subscribers
            #
            with self.cache.lock_key(store_name='subscribers', key=topic):
                subscribers = self.cache.get_kv(store_name='subscribers', key=topic, restore=False).result()
                if subscribers is not None:
                    if self.uid in subscribers:
                        subscribers.remove(self.uid)
                    self.cache.set_kv(store_name='subscribers', key=topic, value=subscribers, persist=False)

            # finally remove my sub object
            #
            del self.subscriptions[topic]

    def publish(self, topic: str, msg: PubSubMsgType) -> bool:
        """
        method to publish a message
        :param topic: the topic of the message
        :param msg: the message to publish
        :return: True if published else False
        """

        if topic not in self.publications:
            self.publications[topic] = {'pub': Pub(topic),
                                        'cache': []}

        # grab the next message id for this topic
        #
        with self.cache.lock_key(store_name='topics', key=topic):
            topic_id = self.cache.get_kv(store_name='topics', key=topic, restore=True).result()
            if topic_id is None:
                topic_id = 0
            else:
                topic_id += 1
            self.cache.set_kv(store_name='topics', key=topic, value=topic_id, persist=True)

        # wrap the message with metadata
        #
        data_packet = {'publisher': self.uid,
                       'timestamp': time.time(),
                       'topic': topic,
                       'msg_id': topic_id,
                       'msg': msg}

        # see if there are subscribers
        #
        with self.cache.lock_key(store_name='subscribers', key=topic):
            subscribers = self.cache.get_kv(store_name='subscribers', key=topic, restore=False).result()

        if subscribers is not None and len(subscribers) > 0:
            # clear any back log in cache
            #
            while len(self.publications[topic]['cache']) > 0:
                msg = self.publications[topic]['cache'].pop(0)
                self.publications[topic]['pub'].put(msg)

                with self.cache.lock_key(store_name='published', key=topic):
                    self.cache.set_kv(store_name='published', key='{}:{}'.format(msg['topic'], msg['msg_id']), value=msg)

            # then publish this message
            #
            self.publications[topic]['pub'].put(data_packet)
            published_key = '{}:{}'.format(data_packet['topic'], data_packet['msg_id'])
            with self.cache.lock_key(store_name='published', key=published_key):
                self.cache.set_kv(store_name='published', key=published_key, value=data_packet)
            published = True
        else:
            # no subscribers so cache
            #
            self.publications[topic]['cache'].append(data_packet)
            published = False

        return published

    def listen(self, timeout: float = None) -> int:
        """
        method to listen for subscribed messages - if any arrive then the associated callback function will be called

        :param timeout: If None the will wait indefinitely, else will timeout within specified seconds
        :return: the number of processed messages
        """
        start_end = time.time()
        nos_processed = 0
        while True:
            for topic in self.subscriptions:
                try:
                    msg = self.subscriptions[topic]['sub'].get(timeout=0.01)
                    self.subscriptions[topic]['callback'](msg)
                    processed_key = '{}:{}:{}'.format(msg['topic'], msg['msg_id'], self.uid)
                    msg['subscriber'] = self.uid
                    msg['topic_msg_id'] = '{}:{}'.format(msg['topic'], msg['msg_id'])
                    self.cache.set_kv(store_name='subscribed', key=processed_key, value=msg)
                    nos_processed += 1
                except Exception as e:
                    time.sleep(0.1)
            if timeout is not None and time.time() - start_end > timeout:
                break

        return nos_processed
