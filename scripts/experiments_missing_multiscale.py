#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from median_consensus_embedding import geometric_median_matrices, normalize_embedding
from utils_illustration import (
    get_dataset,
    mds_from_distance,
    compute_distance_matrix_embedding,
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
    "N_EXP_A_REPEATS": 20,
    "PERPLEXITY_A": 30,
    "PERPLEXITIES_B": [10, 30, 90, 270],
    "N_RUNS_B_PER_PERP": 20,
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


def run_experiment_B_perplexity(X_df, labels):
    print("\nExperiment B: Multiscale Consensus")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    perplexities = CONFIG["PERPLEXITIES_B"]
    all_embeddings_flat = []
    embeddings_by_perp = {}
    perp_representatives = {}

    for perp in perplexities:
        print(f"Running Perplexity = {perp}")
        embeddings_this_perp = []

        for i in tqdm(range(CONFIG["N_RUNS_B_PER_PERP"]), leave=False):
            seed = perp * 1000 + i
            emb = run_tsne(X_scaled, perplexity=perp, random_state=seed)
            embeddings_this_perp.append(emb)
            all_embeddings_flat.append(emb)

        embeddings_by_perp[perp] = embeddings_this_perp
        perp_representatives[perp] = embeddings_this_perp[0]

    dist_mats = [compute_distance_matrix_embedding(e) for e in all_embeddings_flat]
    D_consensus_final = geometric_median_matrices(dist_mats)

    Y_final = mds_from_distance(D_consensus_final, random_state=42)

    perp_distances = {p: [] for p in perplexities}
    for perp in perplexities:
        for emb in embeddings_by_perp[perp]:
            D_ind = compute_distance_matrix_embedding(emb)
            dist = np.linalg.norm(D_consensus_final - D_ind, ord="fro")
            perp_distances[perp].append(dist)

    print("\nSummary Statistics (Distance to Final Consensus)")
    print(f"{'Perplexity':<15} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 55)
    for p in perplexities:
        d = perp_distances[p]
        print(
            f"{p:<15} | {np.mean(d):.4f}   | {np.std(d):.4f}   | {np.min(d):.4f}   | {np.max(d):.4f}"
        )

    plot_scatter_with_legend(
        Y_final, labels, "ExpB_Final_Consensus.pdf", CONFIG["SAVE_PDF"]
    )

    for perp in perplexities:
        title = f"Representative t-SNE (Perplexity={perp})"
        fname = f"ExpB_Representative_Perp{perp}.pdf"
        plot_scatter_with_legend(
            perp_representatives[perp], labels, fname, CONFIG["SAVE_PDF"], ref_Y=Y_final
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
