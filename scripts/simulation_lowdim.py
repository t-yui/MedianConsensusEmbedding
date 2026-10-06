#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from median_consensus_embedding import normalize_embedding
from utils_illustration import (
    compute_distance_matrix_embedding,
    consensus_distance,
    mds_from_distance,
    mean_pairwise_distance,
    mean_distance_to_target,
    structure_recovery,
)


CONFIG = {
    "SCENARIOS": {
        "local": {"sigma_L": 0.025, "sigma_G": 0.0, "pi": 0.0},
        "global": {"sigma_L": 0.01, "sigma_G": 0.075, "pi": 0.0},
        "failure": {"sigma_L": 0.025, "sigma_G": 0.075, "pi": 0.2},
    },
    "N_PER_CLUSTER": 30,
    "SUBCLUSTER_SPACING": 0.25,
    "WITHIN_SD": 0.05,
    "METHODS": ["mode", "vm2012", "mce"],
    "N_CLUSTERS": 9,
    "N_REF": 1000,
    "M_LIST": [5, 10, 20],
    "N_EXEC": 10,
    "N_REPEATS": 50,
    "N_JOBS": 4,
    "RANDOM_STATE": 100,
    "RESULT_DIR": "./results_simulation",
}


# true configuration and synthetic embeddings

def make_truth(rng):
    angles = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    super_centers = directions / np.sqrt(3)
    centers = np.array(
        [
            c + CONFIG["SUBCLUSTER_SPACING"] / np.sqrt(3) * d
            for c in super_centers
            for d in directions
        ]
    )
    groups = np.repeat(np.arange(len(centers)), CONFIG["N_PER_CLUSTER"])
    eps = rng.normal(scale=CONFIG["WITHIN_SD"], size=(len(groups), 2))
    Z = centers[groups] + eps
    return {"Z": Z, "centers": centers, "groups": groups, "eps": eps}


def generate_embedding(truth, sigma_L, sigma_G, pi, rng):
    n_groups = len(truth["centers"])
    if rng.random() < pi:
        mu = rng.uniform(
            truth["Z"].min(axis=0), truth["Z"].max(axis=0), size=(n_groups, 2)
        )
    else:
        mu = truth["centers"] + rng.normal(scale=sigma_G, size=(n_groups, 2))
    Y = (
        mu[truth["groups"]]
        + truth["eps"]
        + rng.normal(scale=sigma_L, size=truth["eps"].shape)
    )
    return normalize_embedding(Y)


# simulation

def compute_targets(truth, params, rng):
    ref = [generate_embedding(truth, **params, rng=rng) for _ in range(CONFIG["N_REF"])]
    return {
        method: consensus_distance(ref, method=method, n_clusters=CONFIG["N_CLUSTERS"])
        for method in CONFIG["METHODS"]
    }


def run_repeat(truth, scenario, params, targets, repeat):
    rng = np.random.default_rng(
        [CONFIG["RANDOM_STATE"], list(CONFIG["SCENARIOS"]).index(scenario), 1, repeat]
    )
    m_max = max(CONFIG["M_LIST"])
    keys = [(1, "single")] + [
        (m, method) for m in CONFIG["M_LIST"] for method in CONFIG["METHODS"]
    ]
    dists = {key: [] for key in keys}
    recoveries = {key: [] for key in keys}

    for e in range(CONFIG["N_EXEC"]):
        runs = [generate_embedding(truth, **params, rng=rng) for _ in range(m_max)]
        mds_seed = repeat * CONFIG["N_EXEC"] + e

        dists[(1, "single")].append(compute_distance_matrix_embedding(runs[0]))
        recoveries[(1, "single")].append(structure_recovery(truth["Z"], runs[0]))

        for m in CONFIG["M_LIST"]:
            for method in CONFIG["METHODS"]:
                D = consensus_distance(
                    runs[:m], method=method, n_clusters=CONFIG["N_CLUSTERS"]
                )
                Y = mds_from_distance(D, random_state=mds_seed)
                dists[(m, method)].append(D)
                recoveries[(m, method)].append(structure_recovery(truth["Z"], Y))

    rows = []
    for m, method in keys:
        target = targets["mce"] if method == "single" else targets[method]
        rec = pd.DataFrame(recoveries[(m, method)]).mean()
        rows.append(
            {
                "scenario": scenario,
                "repeat": repeat,
                "m": m,
                "method": method,
                "s_pair": mean_pairwise_distance(dists[(m, method)]),
                "s_true": mean_distance_to_target(dists[(m, method)], target),
                "q_local": rec["q_local"],
                "q_global": rec["q_global"],
                "auc_rnx": rec["auc_rnx"],
            }
        )
    return rows


if __name__ == "__main__":
    os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
    truth = make_truth(np.random.default_rng([CONFIG["RANDOM_STATE"], 0]))

    rows = []
    for s, (scenario, params) in enumerate(CONFIG["SCENARIOS"].items()):
        print(f"Scenario: {scenario}")
        targets = compute_targets(
            truth, params, np.random.default_rng([CONFIG["RANDOM_STATE"], s, 0])
        )
        results = Parallel(n_jobs=CONFIG["N_JOBS"], verbose=10)(
            delayed(run_repeat)(truth, scenario, params, targets, repeat)
            for repeat in range(CONFIG["N_REPEATS"])
        )
        rows.extend(row for result in results for row in result)

    raw = pd.DataFrame(rows)
    metrics = ["s_pair", "s_true", "q_local", "q_global", "auc_rnx"]
    summary = (
        raw.groupby(["scenario", "m", "method"], sort=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns
    ]

    raw.to_csv(os.path.join(CONFIG["RESULT_DIR"], "lowdim_raw.csv"), index=False)
    summary.to_csv(os.path.join(CONFIG["RESULT_DIR"], "lowdim_summary.csv"), index=False)
    print(summary[summary["m"].isin([1, 10])].to_string(index=False))
