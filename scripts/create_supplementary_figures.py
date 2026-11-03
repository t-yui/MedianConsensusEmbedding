#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import os

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from simulation_lowdim import make_truth, generate_embedding, CONFIG as LOWDIM_CONFIG
from simulation_highdim import make_data, run_base, CONFIG as HIGHDIM_CONFIG
 
 
CONFIG = {
    "RANDOM_STATE": 0,
    "N_INFORMATIVE_IN_SUBSET": [2, 1, 0],
    "RESULT_DIR": "./results_simulation",
}
 
 
def plot_panel(ax, Y, Z, groups, title):
    _, Y, _ = procrustes(Z, Y)
    ax.scatter(Y[:, 0], Y[:, 1], c=groups, cmap="tab10", vmin=0, vmax=9, s=6)
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_lowdim(truth, rng):
    Z, groups = truth["Z"], truth["groups"]
    scenarios = LOWDIM_CONFIG["SCENARIOS"]
    panels = [
        ("(a) True configuration", Z),
        ("(b) L1", generate_embedding(truth, **scenarios["local"], rng=rng)),
        ("(c) L2", generate_embedding(truth, **scenarios["global"], rng=rng)),
        (
            "(d) L3 (failed run)",
            generate_embedding(truth, **{**scenarios["failure"], "pi": 1.0}, rng=rng),
        ),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, Y) in zip(axes, panels):
        plot_panel(ax, Y, Z, groups, title)
    fig.savefig("sim_example_lowdim.pdf", bbox_inches="tight")
    print("Saved: sim_example_lowdim.pdf")
 
 
def draw_subset(rng, n_informative):
    # subsets of the sparse data with the given number of informative covariates
    while True:
        subset = rng.choice(
            HIGHDIM_CONFIG["N_FEATURES"], HIGHDIM_CONFIG["SUBSET_SIZE"], replace=False
        )
        if np.sum(subset < HIGHDIM_CONFIG["N_INFORMATIVE"]) == n_informative:
            return subset
 
 
def plot_highdim(truth, data, rng):
    Z, groups = truth["Z"], truth["groups"]
    columns = [("(a) H1", data["all"], None), ("(b) H2", data["sparse"], None)] + [
        (f"({c}) H3 ({k} informative)", data["sparse"], draw_subset(rng, k))
        for c, k in zip("cde", CONFIG["N_INFORMATIVE_IN_SUBSET"])
    ]
    seeds = rng.integers(0, 2**31 - 1, len(columns))
    bases = [("tsne", "$t$-SNE"), ("umap", "UMAP")]
    fig, axes = plt.subplots(len(bases), len(columns), figsize=(20, 8))
    for i, (base, name) in enumerate(bases):
        for j, ((title, H, subset), seed) in enumerate(zip(columns, seeds)):
            Y = run_base(H, base, seed, subset)
            plot_panel(axes[i, j], Y, Z, groups, title if i == 0 else "")
        axes[i, 0].set_ylabel(name, fontsize=14)
    fig.savefig("sim_example_highdim.pdf", bbox_inches="tight")
    print("Saved: sim_example_highdim.pdf")
 
 
if __name__ == "__main__":
    truth = make_truth(np.random.default_rng([LOWDIM_CONFIG["RANDOM_STATE"], 0]))
    data = make_data(
        truth["Z"], np.random.default_rng([HIGHDIM_CONFIG["RANDOM_STATE"], 0])
    )
 
    os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
    os.chdir(CONFIG["RESULT_DIR"])
 
    rng = np.random.default_rng(CONFIG["RANDOM_STATE"])
    plot_lowdim(truth, rng)
    plot_highdim(truth, data, rng)
