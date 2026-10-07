#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils_illustration import (
    get_dataset,
    run_dr_method,
    run_lle,
    compute_distance_matrix_embedding,
    consensus_distance,
    mds_from_distance,
    structure_recovery,
    plot_scatter_with_legend,
)


CONFIG = {
    "DATA_SOURCE": "toxo",  # 'toxo' or 'eb'
    "METHOD": "tsne",  # 'tsne' or 'umap',
    "CONSENSUS_METHODS": ["vm2012", "mce"],
    "LLE_NEIGHBORS": range(5, 31),
    "N_RUNS_BASE": 1000,
    "N_EVAL": 10,
    "RUNS_LIST": [2, 10, 20, 50, 100],
    "RANDOM_STATE": 0,
    "SAVE_PDF": True,
    "TOXO_FILES": {
        "data": "./data_Barylyuk2020ToxoLopit.csv",
        "label": "./label_Barylyuk2020ToxoLopit.csv",
    },
    "EB_FILES": {"path": "./EBdata.mat", "sample_ratio": 0.1},
}

METHOD_LABELS = {"single": "Single run", "vm2012": "VM2012", "mce": "MCE", "clle": "C-LLE"}


def summarize(m, method, dists, recoveries, target=None):
    pairwise = [
        np.linalg.norm(a - b, ord="fro") for a, b in itertools.combinations(dists, 2)
    ]
    to_target = (
        [np.linalg.norm(D - target, ord="fro") for D in dists]
        if target is not None
        else [np.nan]
    )
    rec = pd.DataFrame(recoveries)
    row = {
        "m": m,
        "method": method,
        "s_true_mean": np.mean(to_target),
        "s_true_sd": np.std(to_target),
        "s_pair_mean": np.mean(pairwise) if pairwise else np.nan,
        "s_pair_sd": np.std(pairwise) if pairwise else np.nan,
    }
    for col in ["q_local", "q_global", "auc_rnx"]:
        row[f"{col}_mean"] = rec[col].mean()
        row[f"{col}_sd"] = rec[col].std(ddof=0)
    return row


def evaluate_consensus(X, targets, n_clusters):
    rng = np.random.default_rng(CONFIG["RANDOM_STATE"])
    rows = []

    for m in [1] + CONFIG["RUNS_LIST"]:
        print(f"\n--- Evaluating for m={m} ({CONFIG['METHOD']}) ---")
        methods = ["single"] if m == 1 else CONFIG["CONSENSUS_METHODS"]
        dists = {method: [] for method in methods}
        recoveries = {method: [] for method in methods}

        for e in tqdm(range(CONFIG["N_EVAL"])):
            seeds = rng.integers(0, 2**31 - 1, m)
            embs = [
                run_dr_method(X, method=CONFIG["METHOD"], random_state=int(s))
                for s in seeds
            ]
            for method in methods:
                if method == "single":
                    D = compute_distance_matrix_embedding(embs[0])
                    Y = embs[0]
                else:
                    D = consensus_distance(embs, method=method, n_clusters=n_clusters)
                    Y = mds_from_distance(D, random_state=e)
                dists[method].append(D)
                recoveries[method].append(structure_recovery(X, Y))

        for method in methods:
            target = targets["mce"] if method == "single" else targets[method]
            rows.append(summarize(m, method, dists[method], recoveries[method], target))
    return rows


def evaluate_clle(X):
    print("\n--- Evaluating C-LLE ---")
    embs = [
        run_lle(X, k, random_state=CONFIG["RANDOM_STATE"])
        for k in tqdm(CONFIG["LLE_NEIGHBORS"])
    ]
    D = consensus_distance(embs, method="mode")
    recoveries = [
        structure_recovery(X, mds_from_distance(D, random_state=e))
        for e in range(CONFIG["N_EVAL"])
    ]
    return summarize(len(CONFIG["LLE_NEIGHBORS"]), "clle", [], recoveries)


def plot_stability(summary):
    x_vals = [1] + CONFIG["RUNS_LIST"]
    panels = [
        ("s_true", r"Distance to $\hat{y}_{1000}$", "instability_plot_1_distance_to_base"),
        ("s_pair", "Distance to each other", "instability_plot_2_pairwise"),
    ]

    for col, ylabel, name in panels:
        plt.figure(figsize=(12, 6))
        colors = {}
        for method in ["single"] + CONFIG["CONSENSUS_METHODS"]:
            part = summary[summary["method"] == method]
            container = plt.errorbar(
                part["m"],
                part[f"{col}_mean"],
                yerr=part[f"{col}_sd"],
                fmt="-o",
                capsize=5,
                markersize=10,
                linewidth=2,
                label=METHOD_LABELS[method],
            )
            colors[method] = container[0].get_color()

        single = summary[summary["method"] == "single"]
        for method in CONFIG["CONSENSUS_METHODS"]:
            first = summary[summary["method"] == method].sort_values("m")
            plt.plot(
                [single["m"].iloc[0], first["m"].iloc[0]],
                [single[f"{col}_mean"].iloc[0], first[f"{col}_mean"].iloc[0]],
                linestyle=":",
                linewidth=2,
                color=colors["single"],
            )
        plt.xlabel(r"Number of embeddings ($m$)", fontsize=24)
        plt.ylabel(ylabel, fontsize=24)
        plt.xticks(x_vals, fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=18)

        filename = f"{name}_{CONFIG['DATA_SOURCE']}_{CONFIG['METHOD']}.pdf"
        if CONFIG["SAVE_PDF"]:
            plt.savefig(filename, format="pdf", bbox_inches="tight")
            print(f"Saved: {filename}")
        else:
            plt.show()


if __name__ == "__main__":
    print(f"Data: {CONFIG['DATA_SOURCE']}, Method: {CONFIG['METHOD']}")

    X_data, labels = get_dataset(CONFIG)
    n_clusters = len(np.unique(labels))

    base_embeddings = []
    for seed in tqdm(range(CONFIG["N_RUNS_BASE"])):
        emb = run_dr_method(X_data, method=CONFIG["METHOD"], random_state=seed)
        base_embeddings.append(emb)

    targets = {
        method: consensus_distance(base_embeddings, method=method, n_clusters=n_clusters)
        for method in CONFIG["CONSENSUS_METHODS"]
    }

    y_mce_base = mds_from_distance(targets["mce"], random_state=0)
    plot_scatter_with_legend(
        y_mce_base,
        labels,
        filename=f"base_consensus_embedding_{CONFIG['DATA_SOURCE']}_{CONFIG['METHOD']}.pdf",
        save_fig=CONFIG["SAVE_PDF"],
    )

    rows = evaluate_consensus(X_data, targets, n_clusters)
    rows.append(evaluate_clle(X_data))
    summary = pd.DataFrame(rows)
    summary.to_csv(
        f"random_init_summary_{CONFIG['DATA_SOURCE']}_{CONFIG['METHOD']}.csv", index=False
    )
    print(summary.to_string(index=False))

    plot_stability(summary)
