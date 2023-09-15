import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import Sampler


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


class ClusterMarginSampler:
    def __init__(self, distance_threshold: float = 1.0, km: int = 5, k_batch: int = 10):
        assert k_batch < km, "k_batch must be smaller than km"
        self._km = km
        self._k_batch = k_batch
        self._distance_threshold = distance_threshold
        self.seed()

    def seed(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    @property
    def km(self):
        return self._km

    @km.setter
    def km(self, value: int):
        assert value > self.k_batch, "km must be larger than k_batch"
        self._km = int(value)

    @property
    def k_batch(self):
        return self._k_batch

    @k_batch.setter
    def k_batch(self, value: int):
        assert value < self.km, "k_batch must be smaller than km"
        self._k_batch = int(value)

    @property
    def n_pool(self):
        pass

    @property
    def distance_threshold(self):
        return self._distance_threshold

    def query(self, k):
        if self.one_sample_step:
            self.one_sample_step = False
            self.emb_list = self.prepare_emb()
            self.HAC_list = AgglomerativeClustering(n_clusters=None, linkage="average", distance_threshold=0.5).fit(np.vstack((x, y)).T)

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        n = min(k * 10, len(self.Y[idxs_unlabeled]))
        index = self.margin_data(n)
        index = self.round_robin(index, self.HAC_list.labels_, k)
        # print(len(index),len([i for i in index if i in idxs_unlabeled]))
        return index
