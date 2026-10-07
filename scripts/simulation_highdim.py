#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from simulation_lowdim import make_truth, CONFIG as LOWDIM_CONFIG
from utils_illustration import (
    run_dr_method,
    run_lle,
    compute_distance_matrix_embedding,
    consensus_distance,
    mds_from_distance,
    mean_pairwise_distance,
    mean_distance_to_target,
    structure_recovery,
)


CONFIG = {
    "SCENARIOS": {
        "all": {"data": "all", "subset": False},
        "sparse": {"data": "sparse", "subset": False},
        "sparse_subset": {"data": "sparse", "subset": True},
    },
    "N_FEATURES": 30,
    "N_INFORMATIVE": 2,
    "SUBSET_SIZE": 15,
    "TAU": 0.05,
    "RHO": 0.1,
    "BASES": ["tsne", "umap"],
    "METHODS": ["vm2012", "mce"],
    "N_CLUSTERS": 9,
    "LLE_NEIGHBORS": range(5, 31),
    "N_REF": 1000,
    "M_LIST": [5, 10, 20],
    "N_EXEC": 10,
    "N_REPEATS": 50,
    "N_JOBS": 4,
    "RANDOM_STATE": 20261007,
    "RESULT_DIR": "./results_simulation",
}


# high-dimensional data

def make_data(Z, rng):
    s_Z = np.sqrt(np.mean(np.var(Z, axis=0)))
    n = Z.shape[0]

    psi = rng.uniform(0, np.pi, CONFIG["N_FEATURES"])
    W = np.column_stack([np.cos(psi), np.sin(psi)])
    H_all = Z @ W.T + rng.normal(scale=CONFIG["TAU"], size=(n, CONFIG["N_FEATURES"]))

    n_noise = CONFIG["N_FEATURES"] - CONFIG["N_INFORMATIVE"]
    H_sparse = np.column_stack(
        [
            Z + rng.normal(scale=CONFIG["TAU"], size=Z.shape),
            rng.normal(scale=CONFIG["RHO"] * s_Z, size=(n, n_noise)),
        ]
    )
    return {"all": H_all, "sparse": H_sparse}


# embeddings

def draw_runs(rng, n_runs, use_subset):
    seeds = rng.integers(0, 2**31 - 1, n_runs)
    subsets = [
        rng.choice(CONFIG["N_FEATURES"], CONFIG["SUBSET_SIZE"], replace=False)
        if use_subset
        else None
        for _ in range(n_runs)
    ]
    return list(zip(seeds, subsets))


def run_base(H, base, seed, subset):
    X = H if subset is None else H[:, subset]
    return run_dr_method(X, method=base, random_state=int(seed))


# simulation

def compute_targets(H, base, use_subset, rng):
    draws = draw_runs(rng, CONFIG["N_REF"], use_subset)
    ref = Parallel(n_jobs=CONFIG["N_JOBS"])(
        delayed(run_base)(H, base, seed, subset) for seed, subset in draws
    )
    return {
        method: consensus_distance(ref, method=method, n_clusters=CONFIG["N_CLUSTERS"])
        for method in CONFIG["METHODS"]
    }


def run_repeat(H, Z, scenario, s, base, b, use_subset, targets, repeat):
    rng = np.random.default_rng([CONFIG["RANDOM_STATE"], s, b, 1, repeat])
    m_max = max(CONFIG["M_LIST"])
    keys = [(1, "single")] + [
        (m, method) for m in CONFIG["M_LIST"] for method in CONFIG["METHODS"]
    ]
    dists = {key: [] for key in keys}
    recoveries = {key: [] for key in keys}

    for e in range(CONFIG["N_EXEC"]):
        runs = [
            run_base(H, base, seed, subset)
            for seed, subset in draw_runs(rng, m_max, use_subset)
        ]
        mds_seed = repeat * CONFIG["N_EXEC"] + e

        dists[(1, "single")].append(compute_distance_matrix_embedding(runs[0]))
        recoveries[(1, "single")].append(structure_recovery(Z, runs[0]))

        for m in CONFIG["M_LIST"]:
            for method in CONFIG["METHODS"]:
                D = consensus_distance(
                    runs[:m], method=method, n_clusters=CONFIG["N_CLUSTERS"]
                )
                Y = mds_from_distance(D, random_state=mds_seed)
                dists[(m, method)].append(D)
                recoveries[(m, method)].append(structure_recovery(Z, Y))

    rows = []
    for m, method in keys:
        if method == "single":
            s_true = np.nan
        else:
            s_true = mean_distance_to_target(
                dists[(m, method)], targets[method]
            )

        rec = pd.DataFrame(recoveries[(m, method)]).mean()
        rows.append(
            {
                "scenario": scenario,
                "repeat": repeat,
                "m": m,
                "method": method,
                "s_pair": mean_pairwise_distance(dists[(m, method)]),
                "s_true": s_true,
                "q_local": rec["q_local"],
                "q_global": rec["q_global"],
                "auc_rnx": rec["auc_rnx"],
            }
        )
    return rows


def run_clle(D, Z, scenario, repeat):
    recoveries = [
        structure_recovery(
            Z, mds_from_distance(D, random_state=repeat * CONFIG["N_EXEC"] + e)
        )
        for e in range(CONFIG["N_EXEC"])
    ]
    rec = pd.DataFrame(recoveries).mean()
    return {
        "scenario": scenario,
        "base": "lle",
        "repeat": repeat,
        "m": len(CONFIG["LLE_NEIGHBORS"]),
        "method": "clle",
        "s_pair": np.nan,
        "s_true": np.nan,
        "q_local": rec["q_local"],
        "q_global": rec["q_global"],
        "auc_rnx": rec["auc_rnx"],
    }


if __name__ == "__main__":
    os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
    truth = make_truth(np.random.default_rng([LOWDIM_CONFIG["RANDOM_STATE"], 0]))
    Z = truth["Z"]
    data = make_data(Z, np.random.default_rng([CONFIG["RANDOM_STATE"], 0]))

    rows = []
    for s, (scenario, setting) in enumerate(CONFIG["SCENARIOS"].items()):
        H = data[setting["data"]]

        print(f"Scenario: {scenario}, C-LLE")
        lle_embeddings = [
            run_lle(H, k, random_state=CONFIG["RANDOM_STATE"])
            for k in CONFIG["LLE_NEIGHBORS"]
        ]
        D_clle = consensus_distance(lle_embeddings, method="mode")
        rows.extend(
            Parallel(n_jobs=CONFIG["N_JOBS"])(
                delayed(run_clle)(D_clle, Z, scenario, repeat)
                for repeat in range(CONFIG["N_REPEATS"])
            )
        )

        for b, base in enumerate(CONFIG["BASES"]):
            print(f"Scenario: {scenario}, base: {base}")
            targets = compute_targets(
                H,
                base,
                setting["subset"],
                np.random.default_rng([CONFIG["RANDOM_STATE"], s, b, 0]),
            )
            results = Parallel(n_jobs=CONFIG["N_JOBS"], verbose=10)(
                delayed(run_repeat)(
                    H, Z, scenario, s, base, b, setting["subset"], targets, repeat
                )
                for repeat in range(CONFIG["N_REPEATS"])
            )
            rows.extend(row for result in results for row in result)

    raw = pd.DataFrame(rows)
    metrics = ["s_pair", "s_true", "q_local", "q_global", "auc_rnx"]
    summary = (
        raw.groupby(["scenario", "base", "m", "method"], sort=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns
    ]

    raw.to_csv(os.path.join(CONFIG["RESULT_DIR"], "highdim_raw.csv"), index=False)
    summary.to_csv(os.path.join(CONFIG["RESULT_DIR"], "highdim_summary.csv"), index=False)
    print(summary[summary["m"].isin([1, 10, 26])].to_string(index=False))
