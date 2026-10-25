#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.spatial import procrustes
from scipy.spatial.distance import squareform
import umap
from median_consensus_embedding import geometric_median_matrices, normalize_embedding


# utils for pre-processing data

def load_data_toxo(data_file, label_file):
    """
    loading and pre-processing TL data
    """

    data_df = pd.read_csv(data_file, index_col=0)
    label_df = pd.read_csv(label_file, index_col=0)

    common_proteins = data_df.index.intersection(label_df.index)
    data_df = data_df.loc[common_proteins]
    label_df = label_df.loc[common_proteins]

    marker_mask = label_df["markers"] != "unknown"
    data_df = data_df[marker_mask]
    label_df = label_df[marker_mask]

    data_normalized = data_df.div(data_df.sum(axis=1), axis=0)

    scaler = StandardScaler()
    X = scaler.fit_transform(data_normalized)
    labels = label_df["markers"].values

    print(f"Loaded: {X.shape[0]} proteins, {X.shape[1]} fractions.")
    return X, labels


def load_data_eb(file_path, sample_ratio=0.1):
    """
    loading and pre-processing EB data
    """
    mat_data = scipy.io.loadmat(file_path)
    data = mat_data["data"]
    labels = mat_data.get("cells", None)
    if labels is not None:
        labels = labels.flatten()

    # down sampling
    n_total = data.shape[0]
    n_sample = int(n_total * sample_ratio)

    print(f"Downsampling data: {n_total} -> {n_sample} cells ({sample_ratio*100}%)")

    np.random.seed(1)
    indices = np.random.choice(n_total, n_sample, replace=False)

    data_sampled = data[indices]

    if labels is not None:
        labels_sampled = labels[indices]
    else:
        labels_sampled = np.zeros(n_sample)

    X = np.sqrt(data_sampled)

    label_dict = {
        1: "00--03 days",
        2: "06--09 days",
        3: "12--15 days",
        4: "18--21 days",
        5: "24--27 days",
    }
    labels_days = np.array([label_dict[e] for e in labels_sampled])

    return X, labels_days


def get_dataset(config):
    if config["DATA_SOURCE"] == "toxo":
        return load_data_toxo(
            config["TOXO_FILES"]["data"], config["TOXO_FILES"]["label"]
        )
    elif config["DATA_SOURCE"] == "eb":
        return load_data_eb(
            config["EB_FILES"]["path"], config["EB_FILES"]["sample_ratio"]
        )


# utils for computation and visualization

def compute_distance_matrix_embedding(Y):
    return pairwise_distances(Y, metric="euclidean")


def run_dr_method(X, method="tsne", random_state=None):
    if method == "tsne":
        model = TSNE(
            n_components=2,
            perplexity=30,
            random_state=random_state,
            init="random",
            learning_rate="auto",
        )
        emb = model.fit_transform(X)
    elif method == "umap":
        model = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            metric="euclidean",
            learning_rate=1,
            init="random",
            min_dist=0.1,
            random_state=random_state,
            n_jobs=1,
        )
        emb = model.fit_transform(X)
    return normalize_embedding(emb)


def run_lle(X, n_neighbors, random_state=None):
    model = LocallyLinearEmbedding(
        n_components=2,
        n_neighbors=n_neighbors,
        method="standard",
        eigen_solver="arpack",
        random_state=random_state,
    )
    emb = model.fit_transform(X)
    return normalize_embedding(emb)


def build_consensus_distance(embeddings_list):
    dist_mats = [compute_distance_matrix_embedding(e) for e in embeddings_list]
    return geometric_median_matrices(dist_mats)


def element_wise_mode(dist_mats, n_grid=128):
    """
    mode of the Gaussian KDE for each pairwise distance (C-LLE type)
    """
    V = np.array([squareform(D, checks=False) for D in dist_mats])
    m = V.shape[0]
    lower = V.min(axis=0)
    upper = V.max(axis=0)
    bandwidth = np.maximum(V.std(axis=0, ddof=1) * m ** (-0.2), np.finfo(float).eps)

    best_density = np.full(V.shape[1], -np.inf)
    mode = lower.copy()
    for t in np.linspace(0, 1, n_grid):
        grid = lower + t * (upper - lower)
        density = np.exp(-0.5 * ((grid - V) / bandwidth) ** 2).sum(axis=0)
        update = density > best_density
        best_density[update] = density[update]
        mode[update] = grid[update]
    return squareform(mode)


def r_squared_index(Y, n_clusters, random_state=0):
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit(Y)
    total = np.sum((Y - Y.mean(axis=0)) ** 2)
    return 1 - km.inertia_ / total


def select_embeddings_vm2012(embeddings_list, n_clusters, threshold=0.15, random_state=0):
    rsi = np.array(
        [r_squared_index(e, n_clusters, random_state) for e in embeddings_list]
    )
    return rsi > threshold * rsi.max()


def consensus_distance(embeddings_list, method="mce", n_clusters=None):
    """
    method: "mce", "vm2012", or "mode" (C-LLE type)
    """
    if method == "mce":
        return build_consensus_distance(embeddings_list)
    elif method == "vm2012":
        selected = select_embeddings_vm2012(embeddings_list, n_clusters)
        dist_mats = [
            compute_distance_matrix_embedding(e)
            for e, keep in zip(embeddings_list, selected)
            if keep
        ]
        return np.median(dist_mats, axis=0)
    elif method == "mode":
        dist_mats = [compute_distance_matrix_embedding(e) for e in embeddings_list]
        return element_wise_mode(dist_mats)


def consensus_embedding(
    X,
    method="mce",
    base="tsne",
    n_runs=10,
    n_clusters=None,
    lle_neighbors=range(5, 31),
    random_state=None,
    mds_random_state=0,
):
    """
    method: "mce", "vm2012", "mode", or "clle" (original C-LLE; base is ignored)
    base: "tsne" or "umap"
    """
    if method == "clle":
        embeddings_list = [
            run_lle(X, k, random_state=random_state) for k in lle_neighbors
        ]
        D = consensus_distance(embeddings_list, method="mode")
    else:
        seeds = np.random.default_rng(random_state).integers(0, 2**31 - 1, n_runs)
        embeddings_list = [
            run_dr_method(X, method=base, random_state=int(s)) for s in seeds
        ]
        D = consensus_distance(embeddings_list, method=method, n_clusters=n_clusters)
    Y = mds_from_distance(D, random_state=mds_random_state)
    return Y, D


def mds_from_distance(D, random_state=0):
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        normalized_stress="auto",
    )
    return mds.fit_transform(D)


def plot_scatter_with_legend(Y, labels, filename=None, save_fig=False, ref_Y=None):

    if ref_Y is not None:
        _, Y, _ = procrustes(ref_Y, Y)

    plt.figure(figsize=(6, 6))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label_name in enumerate(unique_labels):
        mask = labels == label_name
        plt.scatter(
            Y[mask, 0], Y[mask, 1], label=label_name, s=20, alpha=0.8, c=[colors[i]]
        )

    plt.xlabel("Dimension 1", fontsize=14)
    plt.ylabel("Dimension 2", fontsize=14)
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    if filename and save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"Saved: {filename}")

    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=10
    )
    if filename and save_fig:
        plt.savefig("legend_" + filename, format="pdf", bbox_inches="tight")
        print(f"Saved: {filename}")
    else:
        plt.show()


# utils for evaluation
 
def mean_pairwise_distance(dist_mats):
    return np.mean(
        [np.linalg.norm(a - b, "fro") for a, b in itertools.combinations(dist_mats, 2)]
    )
 
 
def mean_distance_to_target(dist_mats, target):
    return np.mean([np.linalg.norm(D - target, "fro") for D in dist_mats])
 
 
def rank_matrix(D):
    D = D.copy()
    np.fill_diagonal(D, -np.inf)
    n = D.shape[0]
    order = np.argsort(D, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=int)
    ranks[np.arange(n)[:, None], order] = np.arange(n)
    return ranks
 
 
def structure_recovery(Z, Y):
    """
    Q_local, Q_global, and AUC of R_NX of the embedding Y against the truth Z
    """
    n = Z.shape[0]
    high = rank_matrix(compute_distance_matrix_embedding(Z))
    low = rank_matrix(compute_distance_matrix_embedding(Y))
 
    max_rank = np.maximum(high, low)
    np.fill_diagonal(max_rank, n)
    counts = np.bincount(max_rank.ravel(), minlength=n + 1)
 
    K = np.arange(1, n)
    qnx = np.cumsum(counts)[1:n] / (n * K)
    lcmc = qnx - K / (n - 1)
    k_max = np.argmax(lcmc) + 1
    q_local = np.mean(qnx[:k_max])
    q_global = np.mean(qnx[k_max - 1 :])
 
    K = np.arange(1, n - 1)
    rnx = ((n - 1) * qnx[:-1] - K) / (n - 1 - K)
    auc_rnx = np.sum(rnx / K) / np.sum(1 / K)
 
    return {"q_local": q_local, "q_global": q_global, "auc_rnx": auc_rnx}
