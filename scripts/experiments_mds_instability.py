#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")

import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.manifold import MDS
from median_consensus_embedding import geometric_median_matrices, normalize_embedding
from utils_illustration import (
    get_dataset,
    run_dr_method,
    mds_from_distance,
    compute_distance_matrix_embedding,
)


CONFIG = {
    "DATA_SOURCE": "toxo",  # "toxo" or "eb"
    "TOXO_FILES": {
        "data": "./data_Barylyuk2020ToxoLopit.csv",
        "label": "./label_Barylyuk2020ToxoLopit.csv",
    },
    "EB_FILES": {"path": "./EBdata.mat", "sample_ratio": 0.1},
    "METHOD": "tsne",  # 'tsne' or 'umap')
    "N_RUNS_BASE": 50,
    "PERPLEXITY": 30,
    "N_MDS_TRIALS": 100,
    "N_JOBS": 4,
    "SAVE_PDF": True,
    "RESULT_DIR": "./results_illustration",
}

if __name__ == "__main__":
    dataset_name = CONFIG["DATA_SOURCE"]
    method = CONFIG["METHOD"]
    print("\nExperiment: MDS Instability")

    X, labels = get_dataset(CONFIG)

    base_embeddings = Parallel(n_jobs=CONFIG["N_JOBS"])(
        delayed(run_dr_method)(X, method=CONFIG["METHOD"], random_state=seed)
        for seed in tqdm(range(CONFIG["N_RUNS_BASE"]), desc="Base Runs")
    )
    base_embeddings = [normalize_embedding(emb) for emb in base_embeddings]

    dist_mats = [compute_distance_matrix_embedding(e) for e in base_embeddings]
    D_consensus = geometric_median_matrices(dist_mats)

    mds_embeddings = Parallel(n_jobs=CONFIG["N_JOBS"])(
        delayed(mds_from_distance)(D_consensus, random_state=seed)
        for seed in tqdm(range(CONFIG["N_MDS_TRIALS"]), desc="MDS Trials")
    )

    mds_dist_mats = [compute_distance_matrix_embedding(y) for y in mds_embeddings]
    distances = []
    pairs = []
    for i, j in itertools.combinations(range(len(mds_dist_mats)), 2):
        disp = np.linalg.norm(mds_dist_mats[i] - mds_dist_mats[j], ord="fro")
        distances.append(disp)
        pairs.append((i, j))

    print(f"Mean Distance: {np.mean(distances):.5f}")
    print(f"SD Mean Distance: {np.std(distances):.5f}")

    # save results
    os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
    tag = f"{dataset_name}_{method}"
    pd.DataFrame(
        [{"trial_1": i, "trial_2": j, "distance": d} for (i, j), d in zip(pairs, distances)]
    ).to_csv(
        os.path.join(CONFIG["RESULT_DIR"], f"mds_instability_raw_{tag}.csv"), index=False
    )
    pd.DataFrame(
        [
            {
                "data": dataset_name,
                "method": method,
                "n_runs_base": CONFIG["N_RUNS_BASE"],
                "n_mds_trials": CONFIG["N_MDS_TRIALS"],
                "mean": np.mean(distances),
                "sd": np.std(distances),
            }
        ]
    ).to_csv(
        os.path.join(CONFIG["RESULT_DIR"], f"mds_instability_summary_{tag}.csv"),
        index=False,
    )
