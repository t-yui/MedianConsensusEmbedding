#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from median_consensus_embedding import geometric_median_matrices
from utils_illustration import (
    get_dataset,
    run_dr_method,
    build_consensus_distance,
    mds_from_distance,
    compute_distance_matrix_embedding,
    plot_scatter_with_legend,
)


CONFIG = {
    "DATA_SOURCE": "toxo",  # 'toxo' or 'eb'
    "METHOD": "tsne",  # 'tsne' or 'umap',
    "N_RUNS_BASE": 1000,
    "N_EVAL": 10,
    "RUNS_LIST": [2, 10, 20, 50, 100],
    "SAVE_PDF": True,
    "TOXO_FILES": {
        "data": "./data_Barylyuk2020ToxoLopit.csv",
        "label": "./label_Barylyuk2020ToxoLopit.csv",
    },
    "EB_FILES": {"path": "./EBdata.mat", "sample_ratio": 0.1},
}


def evaluate_instability(X, labels, y_base_mds, consensus_D_base, run_counts):
    results = {}

    for n_runs in run_counts:
        print(f"\n--- Evaluating for N={n_runs} ({CONFIG['METHOD']}) ---")
        instabilities = []
        pairwise_diffs = []

        temp_dist_matrices = []

        for _ in tqdm(range(CONFIG["N_EVAL"])):
            current_embs = []
            _n = 0
            while _n < n_runs:
                try:
                    seed = np.random.randint(0, 2 ** 32 - 1)
                    emb = run_dr_method(X, method=CONFIG["METHOD"], random_state=seed)
                except Exception as e:
                    continue
                current_embs.append(emb)
                _n += 1

            if n_runs == 1:
                D_current = compute_distance_matrix_embedding(current_embs[0])
            else:
                D_current = build_consensus_distance(current_embs)

            temp_dist_matrices.append(D_current)

            instability = np.linalg.norm(consensus_D_base - D_current, ord="fro")
            instabilities.append(instability)

        if len(temp_dist_matrices) > 1:
            diffs = [
                np.linalg.norm(m1 - m2, ord="fro")
                for m1, m2 in itertools.combinations(temp_dist_matrices, 2)
            ]
            pairwise_diffs = diffs
        else:
            pairwise_diffs = [0.0]

        results[n_runs] = {
            "instabilities": instabilities,
            "pairwise_diffs": pairwise_diffs,
        }

        print(f"  Mean: Dist to Base: {np.mean(instabilities):.5f}")
        print(f"  SD: Dist to Base: {np.std(instabilities):.5f}")
        print(f"  Mean Pairwise Dist: {np.mean(pairwise_diffs):.5f}")
        print(f"  SD: Pairwise Dist: {np.std(pairwise_diffs):.5f}")

    return results


def plot_instability_results(results_single, results_multi):
    runs_list = sorted(list(results_multi.keys()))
    x_vals = [1] + runs_list

    means_cons = [np.mean(results_single[1]["instabilities"])]
    stds_cons = [np.std(results_single[1]["instabilities"])]

    means_pair = [np.mean(results_single[1]["pairwise_diffs"])]
    stds_pair = [np.std(results_single[1]["pairwise_diffs"])]

    for n in runs_list:
        means_cons.append(np.mean(results_multi[n]["instabilities"]))
        stds_cons.append(np.std(results_multi[n]["instabilities"]))

        means_pair.append(np.mean(results_multi[n]["pairwise_diffs"]))
        stds_pair.append(np.std(results_multi[n]["pairwise_diffs"]))

    def _plot_errorbar(y_means, y_stds, ylabel, filename):
        plt.figure(figsize=(12, 6))
        plt.errorbar(
            x_vals,
            y_means,
            yerr=y_stds,
            fmt="-o",
            capsize=5,
            ecolor="black",
            markersize=10,
            linewidth=2,
        )
        plt.xlabel(r"Number of embeddings ($m$)", fontsize=24)
        plt.ylabel(ylabel, fontsize=24)
        plt.xticks(x_vals, fontsize=20)
        plt.yticks(fontsize=20)

        if filename and CONFIG["SAVE_PDF"]:
            plt.savefig(filename, format="pdf", bbox_inches="tight")
            print(f"Saved: {filename}")
        else:
            plt.show()

    _plot_errorbar(
        means_cons,
        stds_cons,
        r"Distance to $\hat{y}_{1000}$",
        f"instability_plot_1_distance_to_base_{CONFIG['DATA_SOURCE']}_{CONFIG['METHOD']}.pdf",
    )

    _plot_errorbar(
        means_pair,
        stds_pair,
        "Distance to each other",
        f"instability_plot_2_pairwise_{CONFIG['DATA_SOURCE']}_{CONFIG['METHOD']}.pdf",
    )


if __name__ == "__main__":
    print(f"Data: {CONFIG['DATA_SOURCE']}, Method: {CONFIG['METHOD']}")

    X_data, labels = get_dataset(CONFIG)

    base_embeddings = []
    for seed in tqdm(range(CONFIG["N_RUNS_BASE"])):
        emb = run_dr_method(X_data, method=CONFIG["METHOD"], random_state=seed)
        base_embeddings.append(emb)

    consensus_D_base = build_consensus_distance(base_embeddings)

    y_mce_base = mds_from_distance(consensus_D_base, random_state=0)
    plot_scatter_with_legend(
        y_mce_base,
        labels,
        filename=f"base_consensus_embedding_{CONFIG['DATA_SOURCE']}_{CONFIG['METHOD']}.pdf",
        save_fig=CONFIG["SAVE_PDF"],
    )

    results_single = evaluate_instability(
        X_data, labels, y_mce_base, consensus_D_base, run_counts=[1]
    )

    results_multi = evaluate_instability(
        X_data, labels, y_mce_base, consensus_D_base, run_counts=CONFIG["RUNS_LIST"]
    )

    plot_instability_results(results_single, results_multi)
