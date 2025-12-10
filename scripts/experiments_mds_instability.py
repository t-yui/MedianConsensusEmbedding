#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
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
    "SAVE_PDF": True,
}

if __name__ == "__main__":
    dataset_name = CONFIG["DATA_SOURCE"]
    method = CONFIG["METHOD"]
    print("\nExperiment: MDS Instability")

    X, labels = get_dataset(CONFIG)

    base_embeddings = []

    for seed in tqdm(range(CONFIG["N_RUNS_BASE"]), desc="Base Runs"):
        emb = run_dr_method(X, method=CONFIG["METHOD"], random_state=seed)
        emb = normalize_embedding(emb)
        base_embeddings.append(emb)

    dist_mats = [compute_distance_matrix_embedding(e) for e in base_embeddings]
    D_consensus = geometric_median_matrices(dist_mats)

    mds_embeddings = []
    for seed in tqdm(range(CONFIG["N_MDS_TRIALS"]), desc="MDS Trials"):
        y = mds_from_distance(D_consensus, random_state=seed)
        mds_embeddings.append(y)

    distances = []
    for i in range(1, len(mds_embeddings)):
        D_current_1 = compute_distance_matrix_embedding(mds_embeddings[i])
        for j in range(1, len(mds_embeddings)):
            D_current_2 = compute_distance_matrix_embedding(mds_embeddings[j])
            disp = np.linalg.norm(D_current_1 - D_current_2, ord="fro")
            distances.append(disp)

    print(f"Mean Distance: {np.mean(distances):.5f}")
    print(f"SD Mean Distance: {np.std(distances):.5f}")
