#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.manifold import TSNE
from sklearn.manifold._t_sne import _joint_probabilities, _kl_divergence
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from median_consensus_embedding import normalize_embedding
from utils import (
    get_dataset,
    build_consensus_distance,
    mds_from_distance,
    compute_distance_matrix_embedding,
    structure_recovery,
    plot_scatter_with_legend,
)


CONFIG = {
    "DATA_SOURCE": "toxo",  # 'toxo' or 'eb'
    "TOXO_FILES": {
        "data": "./data_Barylyuk2020ToxoLopit.csv",
        "label": "./label_Barylyuk2020ToxoLopit.csv",
    },
    "EB_FILES": {"path": "./EBdata.mat", "sample_ratio": 0.1},
    "PERPLEXITIES": [10, 30, 90, 270],
    "N_RUNS_PER_PERP": 20,
    "N_RUNS_MULTISCALE": 20,
    "SAVE_PDF": True,
    "N_JOBS": 4,
    "RESULT_DIR": "./results_illustration",
}


def run_tsne(X, perplexity=30, random_state=None):
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )
    emb = model.fit_transform(X)
    return normalize_embedding(emb)


def tsne_cross_entropy(params, P, n_samples):
    kl, grad = _kl_divergence(params, P, 1, n_samples, 2)
    constant = 2.0 * np.dot(P, np.log(np.maximum(P, np.finfo(np.float64).eps)))
    return kl - constant, grad


def run_multiscale_tsne(X, perplexities, random_state=None, max_iter_per_scale=30):
    """
    multiscale t-SNE (Lee et al., 2015) with random initialization
    """
    X = np.asarray(X, dtype=np.float32)
    n_samples = X.shape[0]
    distances = pairwise_distances(X, metric="euclidean", squared=True)
    probabilities = [
        _joint_probabilities(distances, perplexity, False)
        for perplexity in sorted(perplexities)
    ]

    params = np.random.RandomState(random_state).standard_normal(n_samples * 2)
    for first_scale in range(len(probabilities) - 1, -1, -1):
        P = np.mean(probabilities[first_scale:], axis=0)
        params = minimize(
            tsne_cross_entropy,
            params,
            args=(P, n_samples),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": max_iter_per_scale,
                "gtol": 1e-5,
                "ftol": 2.220446049250313e-9,
                "maxls": 30,
                "maxcor": 6,
                "maxfun": np.inf,
            },
        ).x
    return normalize_embedding(params.reshape(n_samples, 2))


# functions for multiscale experiments


def build_multiscale_consensus(X, run_fn, param_name, param_values, n_runs):
    embeddings_by_param = {}
    for value in param_values:
        print(f"Running {param_name} = {value}")
        embeddings_by_param[value] = Parallel(n_jobs=CONFIG["N_JOBS"])(
            delayed(run_fn)(X, value, random_state=value * 1000 + i)
            for i in tqdm(range(n_runs), leave=False)
        )
    all_embeddings = [e for embs in embeddings_by_param.values() for e in embs]
    return embeddings_by_param, build_consensus_distance(all_embeddings)


def print_distance_summary(D_consensus, embeddings_by_param, param_name):
    print("\nSummary Statistics (Distance to Final Consensus)")
    print(f"{param_name:<15} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 55)
    raw_rows = []
    summary_rows = []
    for value, embs in embeddings_by_param.items():
        d = [
            np.linalg.norm(D_consensus - compute_distance_matrix_embedding(e), ord="fro")
            for e in embs
        ]
        print(
            f"{value:<15} | {np.mean(d):.4f}   | {np.std(d):.4f}   | {np.min(d):.4f}   | {np.max(d):.4f}"
        )
        raw_rows += [
            {param_name.lower(): value, "run": i, "distance": di}
            for i, di in enumerate(d)
        ]
        summary_rows.append(
            {
                param_name.lower(): value,
                "mean": np.mean(d),
                "std": np.std(d),
                "min": np.min(d),
                "max": np.max(d),
            }
        )

    pd.DataFrame(raw_rows).to_csv(
        f"multiscale_distance_raw_{CONFIG['DATA_SOURCE']}_tsne.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        f"multiscale_distance_summary_{CONFIG['DATA_SOURCE']}_tsne.csv", index=False
    )


def run_multiscale_experiment(X_df, labels):
    print("\nExperiment: Multiscale Hyperparameters (t-SNE)")

    X_scaled = StandardScaler().fit_transform(X_df)
    embeddings_by_perp, D_consensus_final = build_multiscale_consensus(
        X_scaled,
        run_tsne,
        "perplexity",
        CONFIG["PERPLEXITIES"],
        CONFIG["N_RUNS_PER_PERP"],
    )
    Y_final = mds_from_distance(D_consensus_final, random_state=42)
    print_distance_summary(D_consensus_final, embeddings_by_perp, "Perplexity")

    print("Running multiscale t-SNE")
    multiscale_embeddings = Parallel(n_jobs=CONFIG["N_JOBS"])(
        delayed(run_multiscale_tsne)(X_scaled, CONFIG["PERPLEXITIES"], random_state=i)
        for i in tqdm(range(CONFIG["N_RUNS_MULTISCALE"]), leave=False)
    )

    # structure recovery
    rows = []
    for perp, embs in embeddings_by_perp.items():
        rows += [
            {"method": f"t-SNE (perplexity={perp})", **structure_recovery(X_scaled, e)}
            for e in embs
        ]
    rows += [
        {"method": "Multiscale t-SNE", **structure_recovery(X_scaled, e)}
        for e in multiscale_embeddings
    ]
    rows.append(
        {"method": "MCE (multi-perplexity t-SNE)", **structure_recovery(X_scaled, Y_final)}
    )
    pd.DataFrame(rows).to_csv(
        f"multiscale_structure_recovery_raw_{CONFIG['DATA_SOURCE']}_tsne.csv", index=False
    )
    recovery = pd.DataFrame(rows).groupby("method", sort=False).agg(["mean", "std"])
    recovery.columns = ["_".join(c) for c in recovery.columns]
    recovery.to_csv(
        f"multiscale_structure_recovery_summary_{CONFIG['DATA_SOURCE']}_tsne.csv"
    )
    print(recovery.to_string())

    plot_scatter_with_legend(
        Y_final, labels, f"multiscale_consensus_embedding_{CONFIG['DATA_SOURCE']}_tsne.pdf", CONFIG["SAVE_PDF"]
    )

    for perp, embs in embeddings_by_perp.items():
        fname = f"multiscale_embedding_perplexity{perp}_{CONFIG['DATA_SOURCE']}_tsne.pdf"
        plot_scatter_with_legend(
            embs[0], labels, fname, CONFIG["SAVE_PDF"], ref_Y=Y_final
        )

    plot_scatter_with_legend(
        multiscale_embeddings[0],
        labels,
        f"multiscale_mstsne_embedding_{CONFIG['DATA_SOURCE']}_tsne.pdf",
        CONFIG["SAVE_PDF"],
        ref_Y=Y_final,
    )


if __name__ == "__main__":
    X_df, labels = get_dataset(CONFIG)

    os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
    os.chdir(CONFIG["RESULT_DIR"])

    run_multiscale_experiment(X_df, labels)
