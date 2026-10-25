#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import experiments_random_init as original
from experiments_random_init import (
    AGGREGATORS,
    POOLS,
    consensus_distance,
    evaluate_clle,
    evaluate_consensus,
    get_dataset,
    mds_from_distance,
    plot_scatter_with_legend,
    plot_stability,
    run_pool,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "DATA_SOURCE": "eb",
    "METHOD": "umap",
    "CONSENSUS_METHODS": ["vm2012", "mce"],
    "SUPPLEMENTARY_METHODS": ["vm2012_full"],
    "SUBSET_FRACTION": 0.5,
    "LLE_NEIGHBORS": range(5, 31),
    "N_RUNS_BASE": 1000,
    "N_EVAL": 10,
    "RUNS_LIST": [2, 10, 20, 50, 100],
    "RANDOM_STATE": 0,
    "SAVE_PDF": True,
    "N_JOBS": 4,
    "TOXO_FILES": {
        "data": str(ROOT / "data_Barylyuk2020ToxoLopit.csv"),
        "label": str(ROOT / "label_Barylyuk2020ToxoLopit.csv"),
    },
    "EB_FILES": {"path": str(ROOT / "EBdata.mat"), "sample_ratio": 0.1},
    "RESULT_DIR": str(ROOT / "results_illustration"),
}


if __name__ == "__main__":
    original.CONFIG.clear()
    original.CONFIG.update(CONFIG)
    os.chdir(ROOT)

    print(f"Data: {CONFIG['DATA_SOURCE']}, Method: {CONFIG['METHOD']}")

    X_data, labels = get_dataset(CONFIG)
    n_clusters = len(np.unique(labels))

    # save all outputs (csv and pdf) in RESULT_DIR
    os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
    os.chdir(CONFIG["RESULT_DIR"])

    base_rng = np.random.default_rng(CONFIG["RANDOM_STATE"] + 1)
    base_pools = {
        pool: run_pool(X_data, pool, tqdm(range(CONFIG["N_RUNS_BASE"])), base_rng)
        for pool in ["full", "subset"]
    }

    targets = {
        method: consensus_distance(
            base_pools[POOLS[method]],
            method=AGGREGATORS[method],
            n_clusters=n_clusters,
        )
        for method in CONFIG["CONSENSUS_METHODS"] + CONFIG["SUPPLEMENTARY_METHODS"]
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
        f"random_init_summary_{CONFIG['DATA_SOURCE']}_{CONFIG['METHOD']}.csv",
        index=False,
    )
    print(summary.to_string(index=False))

    plot_stability(summary)
