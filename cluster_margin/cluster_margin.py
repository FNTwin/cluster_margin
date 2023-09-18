import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import Sampler
from cluster_margin.utils import random_sample
from random import shuffle 
from loguru import logger 
from typing import Optional

class BaseLearner(ABC):
    def fit(self, **fit_kwargs):
        raise NotImplementedError

    def predict(self, **predict_kwargs):
        raise NotImplementedError

    def query(self, **query_kwargs):
        raise NotImplementedError

    def teach(self, **teach_kwargs):
        raise NotImplementedError

    def predict_proba(self, **predict_proba_kwargs):
        raise NotImplementedError
    


class TorchLeaner(BaseLearner):
    pass
     


class ClusterMargin:
    def __init__(self,
                pool,
                model : BaseLearner,
                labelled_sample_size : int = 30, 
                distance_threshold: float = 1.0,
                km: int = 10,
                k_batch: int = 5, 
                iterations : int = 10,
                seed : int = 42,
                verbose : bool = False):
        assert k_batch < km, "k_batch must be smaller than km"
        self.pool = pool
        self.model = model 
        self.number_of_samples = len(pool)
        self.p = labelled_sample_size
        self._km = km
        self._k_batch = k_batch
        self._distance_threshold = distance_threshold
        self._first_iteration = True 
        self._current_iteration = 0
        self.max_iterations = iterations
        self.verbose = verbose
        self.seed(seed)

    def seed(self, seed=42):
        if self.verbose: logger.info(f"Setting seed to {seed}")
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def prepare_embedding(self):
        # add logic for produce embedding from model 
        self.clustering = AgglomerativeClustering(n_clusters=None, linkage="average",
                                                   distance_threshold=self.distance_threshold).fit(self.pool)

    def _initialize_embedding(self):
        # initialize the embedding must have the model embedding implemented, this is for a Dummy model already prepared
        self.training_pool , self.labelled_idxs = random_sample(self.pool, self.p)
        if self.verbose: logger.info("Initializing embedding, randomly selected idxs : {}".format(self.labelled_idxs))
        self.prepare_embedding()

    def get_unlabelled(self):
        all_idxs = np.arange(self.number_of_samples)
        mask = np.ones(len(all_idxs), dtype=bool)
        mask[self.labelled_idxs] = False
        self.unlabelled_idxs  = all_idxs[mask]
        return self.unlabelled_idxs
    
    def predict_proba(self, idxs):
        if self.verbose: logger.info("Predicting probabilities for idxs {}".format(idxs))
        probs = self.model.predict_proba(len(idxs))
        return probs
    

    def get_M(self):
        probs = self.predict_proba(self.unlabelled_idxs)
        if self.verbose: logger.info("Predicted probabilities for unlabelled idxs {} ".format(probs))
        if isinstance(probs, torch.Tensor):
            probs_sorted, idxs = probs.sort(descending=True)
            U = probs_sorted[:, 0] - probs_sorted[:, 1]
            M = self.unlabelled_idxs[U.sort()[1][:self.km]]
        else:
            probs_sorted, idxs = np.sort(probs, axis=1)[:, ::-1]  # invert columns
            U = probs_sorted[:, 0] - probs_sorted[:, 1]
            M =  self.unlabelled_idxs[np.argsort(U)[:self.km]]
        self.M_idx = M
        if self.verbose: logger.info("Querying idxs {}".format(M))
        return M
        
    def _get_sorted_cluster_list(self):

        clust_list = [[] for _ in range(self.clustering.n_clusters_)]
        for idx in self.M_idx:
            cluster_idx = self.clustering.labels_[idx]
            clust_list[cluster_idx].append(idx)
        clust_list.sort(key=lambda x: len(x))
        # remove empty clusters
        clust_list = [x for x in clust_list if x != []]
        self.cluster_list = clust_list
        if self.verbose: logger.info("Sorted cluster list {}".format(clust_list))

    def round_robin_sampling(self):
        if self.verbose: logger.info("Round robin sampling")
        index_select = []
        cluster_index = 0
        k = self.k_batch
        while k > 0:
            if len(self.cluster_list[cluster_index]) > 0:
                shuffle(self.cluster_list[cluster_index])
                index_select.append(self.cluster_list[cluster_index].pop(0))
                k -= 1
            if cluster_index < len(self.cluster_list) - 1:
                cluster_index += 1
            else:
                if len(self.cluster_list[cluster_index]) == 0:  
                    break
                cluster_index = 0
        return index_select
    
    def update(self, idxs):
        self.labelled_idxs = np.concatenate((self.labelled_idxs, idxs))
        if self.verbose: logger.info("New len labelled idxs {}".format(len(self.labelled_idxs)))
    
    def _do_iteration(self):
        if self.verbose: logger.info("Starting iteration {}".format(self._current_iteration))
        if self._first_iteration:
            self._first_iteration = False
            self._initialize_embedding()
        self.get_unlabelled()
        self.get_M()
        self._get_sorted_cluster_list()
        idxs = self.round_robin_sampling()
        return idxs 
    
    def query_iterator(self, batch_size : int = 10, iterations : int = 10):
        while self._current_iteration < iterations:
            self.k_batch = batch_size
            queried_idxs = self._do_iteration()
            self.update(queried_idxs)
            yield queried_idxs
            self._current_iteration += 1


    def query(self, batch_size : Optional[int] = None):
        self.k_batch = batch_size
        queried_idxs = self._do_iteration()
        self.update(queried_idxs)
        self._current_iteration += 1
        return queried_idxs
    
    def get_sorted_cluster_list(clust, set_unlabelled_idxs):
        clust_list = [[] for _ in range(clust.n_clusters_)]
        for idx in set_unlabelled_idxs:
            cluster_idx = clust.labels_[idx]
            clust_list[cluster_idx].append(idx)
        clust_list.sort(key=lambda x: len(x))
        # remove empty clusters
        clust_list = [x for x in clust_list if x != []]
        return clust_list    

    @property
    def km(self):
        return self._km

    @km.setter
    def km(self, value: int):
        if value is not None:
            assert value > self.k_batch, "km must be larger than k_batch"
            self._km = int(value)

    @property
    def k_batch(self):
        return self._k_batch

    @k_batch.setter
    def k_batch(self, value: int):
        if value is not None:
            assert value < self.km, "k_batch must be smaller than km"
            self._k_batch = int(value)

    @property
    def distance_threshold(self):
        return self._distance_threshold
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._current_iteration >= self.max_iterations:
            raise StopIteration
        return self.query(self.k_batch)
    