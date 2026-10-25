#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from median_consensus_embedding import geometric_median_matrices, normalize_embedding
from utils import (
    get_dataset,
    mds_from_distance,
    compute_distance_matrix_embedding,
    plot_scatter_with_legend,
)


CONFIG = {
    "DATA_SOURCE": "toxo",  # 'toxo' or 'eb'
    "TOXO_FILES": {
        "data": "./data_Barylyuk2020ToxoLopit.csv",
        "label": "./label_Barylyuk2020ToxoLopit.csv",
    },
    "EB_FILES": {"path": "./EBdata.mat", "sample_ratio": 0.1},
    "N_RUNS_BASE": 1000,
    "N_IMPUTATIONS": 50,
    "N_REPEATS": 50,
    "PERPLEXITY": 30,
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


# functions for MI experiments


def introduce_missing_values(X, rate, pattern, random_state):
    np.random.seed(random_state)
    X_miss = X.copy()
    n_rows, n_cols = X.shape
    mask = np.zeros_like(X, dtype=bool)

    if pattern == "random":
        mask = np.random.rand(n_rows, n_cols) < rate

    elif pattern == "low_intensity":
        threshold = np.nanpercentile(X, 30)
        prob = np.where(X < threshold, rate * 2.0, rate * 0.5)
        prob = np.clip(prob, 0, 1)
        rand_mat = np.random.rand(n_rows, n_cols)
        mask = rand_mat < prob

    X_miss[mask] = np.nan
    return X_miss


def run_imputation_repeat(X_df, rate, pattern, i, base_consensus_D):
    scaler = StandardScaler()
    X_miss_df = introduce_missing_values(X_df, rate, pattern, random_state=i * 100)
    imp_embeddings = []
    for j in range(CONFIG["N_IMPUTATIONS"]):
        seed = i * 1000 + j
        imputer = IterativeImputer(
            max_iter=10, random_state=seed, sample_posterior=True, verbose=0
        )

        try:
            X_imp = imputer.fit_transform(X_miss_df)
        except:
            from sklearn.impute import SimpleImputer

            X_imp = SimpleImputer().fit_transform(X_miss_df)

        X_imp_scaled = scaler.fit_transform(X_imp)
        emb = run_tsne(
            X_imp_scaled, perplexity=CONFIG["PERPLEXITY"], random_state=seed
        )
        imp_embeddings.append(emb)

    D_imp_consensus = geometric_median_matrices(
        [compute_distance_matrix_embedding(e) for e in imp_embeddings]
    )
    dist = np.linalg.norm(base_consensus_D - D_imp_consensus, ord="fro")
    y_scen = mds_from_distance(D_imp_consensus, random_state=0) if i == 0 else None
    return dist, y_scen


def run_imputation_experiment(X_df, labels, base_consensus_D):
    print("\nExperiment: Multiple Imputation")

    # missing scenarios
    scenarios = [
        (0.1, "random"),
        (0.1, "low_intensity"),
        (0.3, "random"),
        (0.3, "low_intensity"),
    ]

    scenario_distances = {s: [] for s in scenarios}
    scenario_example_plots = {}

    # each repeat (scenario, i) is run in parallel
    tasks = [(s, i) for s in scenarios for i in range(CONFIG["N_REPEATS"])]
    results = Parallel(n_jobs=CONFIG["N_JOBS"], verbose=10)(
        delayed(run_imputation_repeat)(X_df, s[0], s[1], i, base_consensus_D)
        for s, i in tasks
    )
    for (s, i), (dist, y_scen) in zip(tasks, results):
        scenario_distances[s].append(dist)
        if i == 0:
            scenario_example_plots[s] = y_scen

    print("\nSummary Statistics (Distance to Base)")
    print(f"{'Scenario':<25} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 65)
    for s in scenarios:
        d = scenario_distances[s]
        name = f"{s[1]} ({int(s[0]*100)}%)"
        print(
            f"{name:<25} | {np.mean(d):.4f}   | {np.std(d):.4f}   | {np.min(d):.4f}   | {np.max(d):.4f}"
        )

    raw = pd.DataFrame(
        [
            {"rate": s[0], "pattern": s[1], "repeat": i, "distance": d}
            for s in scenarios
            for i, d in enumerate(scenario_distances[s])
        ]
    )
    raw.to_csv(
        f"missing_raw_{CONFIG['DATA_SOURCE']}_tsne.csv", index=False
    )
    summary = (
        raw.groupby(["rate", "pattern"], sort=False)["distance"]
        .agg(
            mean="mean",
            std=lambda x: np.std(x),
            min="min",
            max="max",
        )
        .reset_index()
    )
    summary.to_csv(
        f"missing_summary_{CONFIG['DATA_SOURCE']}_tsne.csv", index=False
    )

    y_base = mds_from_distance(base_consensus_D, random_state=0)
    plot_scatter_with_legend(
        y_base, labels, f"missing_base_consensus_embedding_{CONFIG['DATA_SOURCE']}_tsne.pdf", CONFIG["SAVE_PDF"]
    )

    for s, y_scen in scenario_example_plots.items():
        title = f"Scenario: {s[1]} ({int(s[0]*100)}%) - Imputed Consensus"
        fname = f"missing_consensus_embedding_{s[1]}_{int(s[0]*100)}_{CONFIG['DATA_SOURCE']}_tsne.pdf"
        plot_scatter_with_legend(
            y_scen, labels, fname, CONFIG["SAVE_PDF"], ref_Y=y_base
        )


if __name__ == "__main__":
    X_df, labels = get_dataset(CONFIG)

    os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
    os.chdir(CONFIG["RESULT_DIR"])

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_df)
    base_embeddings = Parallel(n_jobs=CONFIG["N_JOBS"])(
        delayed(run_tsne)(X_full_scaled, perplexity=CONFIG["PERPLEXITY"], random_state=i)
        for i in tqdm(range(CONFIG["N_RUNS_BASE"]))
    )
    base_dist_mats = [compute_distance_matrix_embedding(e) for e in base_embeddings]
    base_consensus_D = geometric_median_matrices(base_dist_mats)

    run_imputation_experiment(X_df, labels, base_consensus_D)
