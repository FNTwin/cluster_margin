from random import shuffle
from typing import Union

import datamol as dm
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from umap import UMAP


def random_sample(pool, n):
    idxs = np.random.choice(np.arange(len(pool)), n, replace=False)
    return pool[idxs], idxs


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    hierarchy.dendrogram(linkage_matrix, **kwargs)


def get_dummy_data():
    # set seed
    np.random.seed(42)
    data = dm.data.chembl_drugs()
    data["mol"] = data["smiles"].apply(dm.to_mol)
    data["fp"] = data["mol"].apply(dm.to_fp)
    umap = UMAP(n_components=2)
    emb_2d = umap.fit_transform(np.vstack(data["fp"].values))
    data["x"] = emb_2d[:, 0]
    data["y"] = emb_2d[:, 1]
    a = torch.rand((len(data), 5))
    a = a / a.sum(dim=1, keepdim=True)
    data["prob0"] = a[:, 0]
    data["prob1"] = a[:, 1]
    data["prob2"] = a[:, 2]
    data["prob3"] = a[:, 3]
    data["prob4"] = a[:, 4]
    return data


def round_robin_sampling(k):
    cluster_list = [["A", "B", "C"], ["D", "E"], ["F"], ["1", "3", "2", "3"], ["KDKDKD"]]
    cluster_list.sort(key=lambda x: len(x))
    index_select = []
    cluster_index = 0
    while k > 0:
        if len(cluster_list[cluster_index]) > 0:
            index_select.append(cluster_list[cluster_index].pop(0))
            k -= 1
        if cluster_index < len(cluster_list) - 1:
            cluster_index += 1
        else:
            if len(cluster_list[cluster_index]) == 0:  # escape condition if k > number of clusters
                break
            cluster_index = 0
    return index_select


def calculate_margin_score(self, probs: Union[torch.tensor, np.ndarray], n: int):
    idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_labeled]
    if isinstance(probs, torch.Tensor):
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:, 1]
        return idxs_unlabeled[U.sort()[1].numpy()[:n]]
    probs_sorted, idxs = np.sort(probs, axis=1)[:, ::-1]  # invert columns
    U = probs_sorted[:, 0] - probs_sorted[:, 1]
    return idxs_unlabeled[np.argsort(U)[:n]]


def initialization_step(pool, n=30):
    labeled_examples, idxs = random_sample(pool.to_numpy(), n)
    x = pool["x"].to_numpy()
    y = pool["y"].to_numpy()
    clust = AgglomerativeClustering(n_clusters=None, linkage="average", distance_threshold=0.5).fit(np.vstack((x, y)).T)
    return {"label_examples": labeled_examples, "label_idxs": idxs, "clustering": clust}


def get_unlabelled(pool, idxs):
    all_idxs = np.arange(len(pool))
    mask = np.ones(len(all_idxs), dtype=bool)
    mask[idxs] = False
    unlabelled_idxs = all_idxs[mask]
    return pool[unlabelled_idxs], unlabelled_idxs


def get_probabilities(data):
    probs = data[["prob0", "prob1", "prob2", "prob3", "prob4"]].to_numpy()
    return probs


def get_mask(pool, idxs):
    # crearte a mask
    all_idxs = np.arange(len(pool))
    mask = np.zeros(len(all_idxs), dtype=bool)
    mask[idxs] = True
    return mask


def get_M(pool, idxs, km=5):
    # get the mask of the selected examples from the margin score criteria
    mask = get_mask(pool, idxs)
    all_probs = get_probabilities(pool)
    probs = all_probs[mask]
    probs_sorted = np.sort(probs, axis=1)[:, ::-1]  # invert columns
    U = probs_sorted[:, 0] - probs_sorted[:, 1]  # margin score
    return np.argsort(U)[:km]


def iteration(pool, n=30, km=30, kt=10):
    assert kt < km, "kt must be smaller than km"
    examples = np.vstack((pool["x"].to_numpy(), pool["y"].to_numpy())).T
    init = initialization_step(pool, n)
    unlabelled_examples, unlabelled_idxs = get_unlabelled(pool.to_numpy(), init["label_idxs"])
    # return examples[get_M(pool, unlabelled_idxs)]
    M_idxs = get_M(pool, unlabelled_idxs, km)
    cluster_list = get_sorted_cluster_list(init["clustering"], M_idxs)
    idxs = round_robin_sampling(cluster_list, kt)
    return init, unlabelled_examples, unlabelled_idxs, examples[get_M(pool, unlabelled_idxs, km)], examples[idxs]


def round_robin_sampling(cluster_list, k):
    index_select = []
    cluster_index = 0
    # on each cluster select one element, then go to the next cluster by size
    # if the cluster is empty, go to the next one
    # if it is the last cluster, go to the first one
    while k > 0:
        if len(cluster_list[cluster_index]) > 0:
            shuffle(cluster_list[cluster_index])
            index_select.append(cluster_list[cluster_index].pop(0))

            # index_select.append(cluster_list[cluster_index].pop(0))
            k -= 1
        if cluster_index < len(cluster_list) - 1:
            cluster_index += 1
        else:
            if len(cluster_list[cluster_index]) == 0:  # escape condition if k > number of clusters
                break
            cluster_index = 0
    return index_select


def get_sorted_cluster_list(clust, set_unlabelled_idxs):
    # create a list for each cluster
    # for every unlabelled idx, append it to the corresponding cluster
    # by finding the cluster index in the clust.labels_ array
    clust_list = [[] for _ in range(clust.n_clusters_)]
    for idx in set_unlabelled_idxs:
        cluster_idx = clust.labels_[idx]
        clust_list[cluster_idx].append(idx)
    clust_list.sort(key=lambda x: len(x))
    # remove empty clusters
    clust_list = [x for x in clust_list if x != []]
    return clust_list


test = iteration(data, 40, km=50, kt=10)
plt.scatter(data["x"].to_numpy()[test[0]["label_idxs"]], data["y"].to_numpy()[test[0]["label_idxs"]], c="red", s=10, label="train")
plt.scatter(data["x"].to_numpy()[test[2]], data["y"].to_numpy()[test[2]], c="black", s=10, label="unlabelled set")
plt.scatter(*test[3].T, color="blue", marker=".", label="margin selected")
plt.scatter(*test[4].T, color="yellow", marker="x", label="batch selected")
plt.legend()


def real_iteration(pool, init, labelled_idxs, n=30, km=30, kt=10):
    unlabelled_examples, unlabelled_idxs = get_unlabelled(pool.to_numpy(), labelled_idxs)
    # return examples[get_M(pool, unlabelled_idxs)]
    M_idxs = get_M(pool, unlabelled_idxs, km)
    cluster_list = get_sorted_cluster_list(init["clustering"], M_idxs)
    idxs = round_robin_sampling(cluster_list, kt)
    return idxs


def cluster_margin(pool, n=30, km=30, kt=10, it=1):
    assert kt < km, "kt must be smaller than km"
    examples = np.vstack((pool["x"].to_numpy(), pool["y"].to_numpy())).T
    init = initialization_step(pool, n)
    tmp = []
    for i in range(it):
        # print(len(init["label_idxs"]))
        new_idxs = real_iteration(pool, init, init["label_idxs"], n, km, kt)
        init["label_idxs"] = np.concatenate((init["label_idxs"], new_idxs))
        tmp.append(examples[new_idxs])
    return tmp
