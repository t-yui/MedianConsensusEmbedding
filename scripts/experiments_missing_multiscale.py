#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.manifold import TSNE, MDS
from sklearn.manifold._t_sne import _joint_probabilities, _kl_divergence
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import umap
from median_consensus_embedding import geometric_median_matrices, normalize_embedding
from utils_illustration import (
    get_dataset,
    build_consensus_distance,
    mds_from_distance,
    compute_distance_matrix_embedding,
    structure_recovery,
    plot_scatter_with_legend,
)


CONFIG = {
    "DATA_SOURCE": "toxo",  # 'toxo' or 'eb'
    "METHOD": "tsne",  # 'tsne' or 'umap',
    "TOXO_FILES": {
        "data": "./data_Barylyuk2020ToxoLopit.csv",
        "label": "./label_Barylyuk2020ToxoLopit.csv",
    },
    "EB_FILES": {"path": "./EBdata.mat", "sample_ratio": 0.1},
    "N_RUNS_BASE": 1000,
    "N_IMPUTATIONS": 50,
    "N_EXP_A_REPEATS": 50,
    "PERPLEXITY_A": 30,
    "PERPLEXITIES_B": [10, 30, 90, 270],
    "N_RUNS_B_PER_PERP": 20,
    "N_RUNS_B_MULTISCALE": 20,
    "N_NEIGHBORS_B": [5, 15, 50, 150],
    "N_RUNS_B_PER_NEIGHBOR": 20,
    "SAVE_PDF": True,
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


def run_umap(X, n_neighbors=15, random_state=None):
    model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        metric="euclidean",
        learning_rate=1,
        init="random",
        min_dist=0.1,
        random_state=random_state,
        n_jobs=1,
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


def run_experiment_A_imputation(X_df, labels, base_consensus_D):
    print("\nExperiment A: Multiple Imputation Consensus")

    # missing scenarios
    scenarios = [
        (0.1, "random"),
        (0.1, "low_intensity"),
        (0.3, "random"),
        (0.3, "low_intensity"),
    ]

    scenario_distances = {s: [] for s in scenarios}
    scenario_example_plots = {}

    scaler = StandardScaler()

    for (rate, pattern) in scenarios:
        print(f"\nProcessing Scenario: Rate={rate}, Pattern={pattern}")

        for i in range(CONFIG["N_EXP_A_REPEATS"]):
            X_miss_df = introduce_missing_values(
                X_df, rate, pattern, random_state=i * 100
            )
            imp_embeddings = []
            pbar = tqdm(
                range(CONFIG["N_IMPUTATIONS"]),
                desc=f"  Iter {i+1}/{CONFIG['N_EXP_A_REPEATS']} Imputing",
                leave=False,
            )

            for j in pbar:
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
                    X_imp_scaled, perplexity=CONFIG["PERPLEXITY_A"], random_state=seed
                )
                imp_embeddings.append(emb)

            D_imp_consensus = geometric_median_matrices(
                [compute_distance_matrix_embedding(e) for e in imp_embeddings]
            )
            dist = np.linalg.norm(base_consensus_D - D_imp_consensus, ord="fro")
            scenario_distances[(rate, pattern)].append(dist)

            if i == 0:
                y_scen = mds_from_distance(D_imp_consensus, random_state=0)
                scenario_example_plots[(rate, pattern)] = y_scen

    # summary statistics
    print("\nSummary Statistics (Distance to Base)")
    print(f"{'Scenario':<25} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 65)
    for s in scenarios:
        d = scenario_distances[s]
        name = f"{s[1]} ({int(s[0]*100)}%)"
        print(
            f"{name:<25} | {np.mean(d):.4f}   | {np.std(d):.4f}   | {np.min(d):.4f}   | {np.max(d):.4f}"
        )

    # visualize example plots
    y_base = mds_from_distance(base_consensus_D, random_state=0)
    plot_scatter_with_legend(
        y_base, labels, "ExpA_Scatter_Base.pdf", CONFIG["SAVE_PDF"]
    )

    for s, y_scen in scenario_example_plots.items():
        title = f"Scenario: {s[1]} ({int(s[0]*100)}%) - Imputed Consensus"
        fname = f"ExpA_Scatter_{s[1]}_{int(s[0]*100)}.pdf"
        plot_scatter_with_legend(
            y_scen, labels, fname, CONFIG["SAVE_PDF"], ref_Y=y_base
        )


# functions for multiscale experiments


def build_multiscale_consensus(X, run_fn, param_name, param_values, n_runs):
    embeddings_by_param = {}
    for value in param_values:
        print(f"Running {param_name} = {value}")
        embeddings_by_param[value] = [
            run_fn(X, value, random_state=value * 1000 + i)
            for i in tqdm(range(n_runs), leave=False)
        ]
    all_embeddings = [e for embs in embeddings_by_param.values() for e in embs]
    return embeddings_by_param, build_consensus_distance(all_embeddings)


def print_distance_summary(D_consensus, embeddings_by_param, param_name):
    print("\nSummary Statistics (Distance to Final Consensus)")
    print(f"{param_name:<15} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 55)
    for value, embs in embeddings_by_param.items():
        d = [
            np.linalg.norm(D_consensus - compute_distance_matrix_embedding(e), ord="fro")
            for e in embs
        ]
        print(
            f"{value:<15} | {np.mean(d):.4f}   | {np.std(d):.4f}   | {np.min(d):.4f}   | {np.max(d):.4f}"
        )


def run_experiment_B_perplexity(X_df, labels):
    print("\nExperiment B: Multiscale Consensus (t-SNE)")

    X_scaled = StandardScaler().fit_transform(X_df)
    embeddings_by_perp, D_consensus_final = build_multiscale_consensus(
        X_scaled,
        run_tsne,
        "perplexity",
        CONFIG["PERPLEXITIES_B"],
        CONFIG["N_RUNS_B_PER_PERP"],
    )
    Y_final = mds_from_distance(D_consensus_final, random_state=42)
    print_distance_summary(D_consensus_final, embeddings_by_perp, "Perplexity")

    print("Running multiscale t-SNE")
    multiscale_embeddings = [
        run_multiscale_tsne(X_scaled, CONFIG["PERPLEXITIES_B"], random_state=i)
        for i in tqdm(range(CONFIG["N_RUNS_B_MULTISCALE"]), leave=False)
    ]

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
    recovery = pd.DataFrame(rows).groupby("method", sort=False).agg(["mean", "std"])
    recovery.columns = ["_".join(c) for c in recovery.columns]
    recovery.to_csv(f"ExpB_structure_recovery_{CONFIG['DATA_SOURCE']}.csv")
    print(recovery.to_string())

    plot_scatter_with_legend(
        Y_final, labels, "ExpB_Final_Consensus.pdf", CONFIG["SAVE_PDF"]
    )

    for perp, embs in embeddings_by_perp.items():
        fname = f"ExpB_Representative_Perp{perp}.pdf"
        plot_scatter_with_legend(
            embs[0], labels, fname, CONFIG["SAVE_PDF"], ref_Y=Y_final
        )

    plot_scatter_with_legend(
        multiscale_embeddings[0],
        labels,
        "ExpB_Multiscale_tSNE.pdf",
        CONFIG["SAVE_PDF"],
        ref_Y=Y_final,
    )


if __name__ == "__main__":
    X_df, labels = get_dataset(CONFIG)

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_df)
    base_embeddings = []
    for i in tqdm(range(CONFIG["N_RUNS_BASE"])):
        emb = run_tsne(X_full_scaled, perplexity=CONFIG["PERPLEXITY_A"], random_state=i)
        base_embeddings.append(emb)
    base_dist_mats = [compute_distance_matrix_embedding(e) for e in base_embeddings]
    base_consensus_D = geometric_median_matrices(base_dist_mats)

    # Experiment A: Multiple Imputation Consensus
    run_experiment_A_imputation(X_df, labels, base_consensus_D)

    # Experiment B: Multiscale Consensus
    run_experiment_B_perplexity(X_df, labels)
