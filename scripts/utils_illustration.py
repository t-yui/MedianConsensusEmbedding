#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.spatial import procrustes
import umap
from median_consensus_embedding import geometric_median_matrices


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


def build_consensus_distance(embeddings_list):
    dist_mats = [compute_distance_matrix_embedding(e) for e in embeddings_list]
    return geometric_median_matrices(dist_mats)


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
