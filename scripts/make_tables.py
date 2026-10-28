#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "SIMULATION_DIR": ROOT / "results_simulation",
    "ILLUSTRATION_DIR": ROOT / "results_illustration",
    "OUTPUT_DIR": ROOT / "results_for_paper",
}

LOWDIM_SCENARIOS = {"local": "L1", "global": "L2", "failure": "L3"}
HIGHDIM_SCENARIOS = {"all": "H1", "sparse": "H2", "sparse_subset": "H3", "all_3d": "H4"}
DATASETS = [("toxo", "tsne", "ToxoLopit"), ("eb", "umap", "Embryoid body")]
M_LIST = [5, 10, 20]
METRICS = ["s_pair", "s_ref", "q_local", "q_global", "auc_rnx"]
METRIC_LABELS = {
    "s_pair": "S_pair",
    "s_ref": "S_ref",
    "q_local": "Q_local",
    "q_global": "Q_global",
    "auc_rnx": "AUC",
}


def fmt(mean, sd=None, digits=3):
    if pd.isna(mean):
        return "--"
    if sd is None or pd.isna(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({sd:.{digits}f})"


def cells(r, metrics, sd="std", s_digits=3):
    # S_pair and S_ref are shown with s_digits, the others with 3 decimals
    return {
        METRIC_LABELS[k]: fmt(
            r[f"{k}_mean"],
            r[f"{k}_{sd}"] if sd else None,
            s_digits if k.startswith("s_") else 3,
        )
        for k in metrics
    }


def get_row(summary, **keys):
    part = summary
    for k, v in keys.items():
        part = part[part[k] == v]
    return part.iloc[0]


def save(df, name):
    path = CONFIG["OUTPUT_DIR"] / name
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def text_row(section, description, mean, sd=None, digits=2):
    text = f"{mean:.{digits}f}"
    if sd is not None:
        text += f" (SD: {sd:.{digits}f})"
    return {"Section": section, "Description": description, "Mean": mean, "SD": sd, "Text": text}


# simulation

def make_simulation_tables():
    lowdim = pd.read_csv(CONFIG["SIMULATION_DIR"] / "lowdim_summary.csv")
    highdim = pd.read_csv(CONFIG["SIMULATION_DIR"] / "highdim_summary.csv")
    clle = highdim[highdim["method"] == "clle"]

    # main text: tab:simulation (m = 10)
    rows = []
    for scenario, name in LOWDIM_SCENARIOS.items():
        part = lowdim[(lowdim["scenario"] == scenario) & (lowdim["m"] == 10)]
        rs = {
            "C-LLE": get_row(part, method="mode"),
            "VM2012": get_row(part, method="vm2012"),
            "MCE": get_row(part, method="mce"),
        }
        rows.append(("Low-dimensional", name, rs))
    for base, group in [("tsne", "High-dimensional with t-SNE"), ("umap", "High-dimensional with UMAP")]:
        for scenario, name in HIGHDIM_SCENARIOS.items():
            part = highdim[
                (highdim["scenario"] == scenario) & (highdim["base"] == base) & (highdim["m"] == 10)
            ]
            rs = {
                "C-LLE": get_row(clle, scenario=scenario),
                "VM2012": get_row(part, method="vm2012"),
                "MCE": get_row(part, method="mce"),
            }
            rows.append((group, name, rs))
    table = []
    for group, name, rs in rows:
        row = {"Group": group, "Scenario": name}
        for k in METRICS:
            for label, r in rs.items():
                row[f"{METRIC_LABELS[k]} {label}"] = cells(r, [k], sd=None, s_digits=2)[METRIC_LABELS[k]]
        table.append(row)
    save(pd.DataFrame(table), "main_tab_simulation.csv")

    # supplementary: tab:sim_lowdim
    labels = {"single": "Single", "mode": "C-LLE-type", "vm2012": "VM2012-type", "mce": "MCE"}
    table = []
    for scenario, name in LOWDIM_SCENARIOS.items():
        part = lowdim[lowdim["scenario"] == scenario]
        keys = [(1, "single")] + [(m, method) for m in M_LIST for method in ["mode", "vm2012", "mce"]]
        for i, (m, method) in enumerate(keys):
            r = get_row(part, m=m, method=method)
            row = {"Scenario": name if i == 0 else "", "m": m, "Method": labels[method]}
            table.append({**row, **cells(r, METRICS)})
    save(pd.DataFrame(table), "supp_tab_sim_lowdim.csv")

    # supplementary: tab:sim_highdim_tsne and tab:sim_highdim_umap
    labels = {"single": "Single", "vm2012": "VM2012", "mce": "MCE"}
    for base in ["tsne", "umap"]:
        table = []
        for scenario, name in HIGHDIM_SCENARIOS.items():
            part = highdim[(highdim["scenario"] == scenario) & (highdim["base"] == base)]
            keys = [(1, "single")] + [(m, method) for m in M_LIST for method in ["vm2012", "mce"]]
            for i, (m, method) in enumerate(keys):
                r = get_row(part, m=m, method=method)
                row = {"Scenario": name if i == 0 else "", "m": m, "Method": labels[method]}
                table.append({**row, **cells(r, METRICS)})
            r = get_row(clle, scenario=scenario)
            table.append({"Scenario": "", "m": "--", "Method": "C-LLE", **cells(r, METRICS)})
        save(pd.DataFrame(table), f"supp_tab_sim_highdim_{base}.csv")


# evaluation of rapid convergence and stability

def make_random_init_tables():
    metrics = ["s_ref", "s_pair", "q_local", "q_global", "auc_rnx"]
    labels = {"vm2012": "VM2012", "vm2012_full": "VM2012all", "mce": "MCE"}
    text = []
    section = "Evaluation of Rapid Convergence and Stability"
    for data, method, name in DATASETS:
        summary = pd.read_csv(
            CONFIG["ILLUSTRATION_DIR"] / f"random_init_summary_{data}_{method}.csv"
        )

        # supplementary: tab:random_init_toxo and tab:random_init_eb
        table = []
        r = get_row(summary, method="single")
        table.append({"m": 1, "Method": "Single (S_ref: y_1000 of MCE)", **cells(r, metrics, "sd", 2)})
        r = get_row(summary, method="single_vm2012")
        row = {k: "" for k in ["m", "Method"] + [METRIC_LABELS[k] for k in metrics]}
        row.update({"Method": "Single (S_ref: y_1000 of VM2012)", **cells(r, ["s_ref"], "sd", 2)})
        table.append(row)
        for m in sorted(summary.loc[summary["method"] == "mce", "m"]):
            for key, label in labels.items():
                r = get_row(summary, m=m, method=key)
                table.append({"m": m, "Method": label, **cells(r, metrics, "sd", 2)})
        r = get_row(summary, method="clle")
        table.append({"m": "--", "Method": "C-LLE", **cells(r, metrics, "sd", 2)})
        save(pd.DataFrame(table), f"supp_tab_random_init_{data}.csv")

        # main text values
        for key, label in [("mce", "MCE"), ("vm2012", "VM2012")]:
            r = get_row(summary, m=10, method=key)
            text.append(text_row(section, f"S_pair of {label} with m=10, {name}", r["s_pair_mean"]))
        for key, label in [("single", "single embedding"), ("mce", "MCE with m=10"), ("vm2012", "VM2012 with m=10"), ("clle", "C-LLE")]:
            r = get_row(summary, method=key) if key in ["single", "clle"] else get_row(summary, m=10, method=key)
            text.append(text_row(section, f"AUC of {label}, {name}", r["auc_rnx_mean"], digits=3))
    return text


# combined approach with multiple imputation

def make_missing_values():
    summary = pd.read_csv(CONFIG["ILLUSTRATION_DIR"] / "missing_summary_toxo_tsne.csv")
    section = "Combined Approach with Multiple Imputation"
    text = []
    for rate in [0.1, 0.3]:
        for pattern, label in [("random", "random"), ("low_intensity", "intensity-dependent")]:
            r = get_row(summary, rate=rate, pattern=pattern)
            text.append(
                text_row(
                    section,
                    f"Distance to y_1000, {label} missingness, {int(rate * 100)}%",
                    r["mean"],
                    r["std"],
                )
            )
    return text


# consensus representation of multiscale hyperparameter

def make_multiscale_tables():
    section = "Consensus Representation of Multiscale Hyperparameter"
    recovery = pd.read_csv(
        CONFIG["ILLUSTRATION_DIR"] / "multiscale_structure_recovery_summary_toxo_tsne.csv"
    )
    recovery["method"] = recovery["method"].replace({"MCE (multi-perplexity t-SNE)": "MCE"})

    # supplementary: tab:multiscale
    table = [
        {"Method": r["method"], **cells(r, ["q_local", "q_global", "auc_rnx"])}
        for _, r in recovery.iterrows()
    ]
    save(pd.DataFrame(table), "supp_tab_multiscale.csv")

    # main text values
    text = []
    distance = pd.read_csv(
        CONFIG["ILLUSTRATION_DIR"] / "multiscale_distance_summary_toxo_tsne.csv"
    )
    for _, r in distance.iterrows():
        text.append(
            text_row(
                section,
                f"Distance from MCE to embeddings with perplexity={int(r['perplexity'])}",
                r["mean"],
                r["std"],
            )
        )
    for method in ["MCE", "Multiscale t-SNE"]:
        r = get_row(recovery, method=method)
        for k in ["q_global", "q_local", "auc_rnx"]:
            text.append(text_row(section, f"{METRIC_LABELS[k]} of {method}", r[f"{k}_mean"], digits=3))
    return text


# instability of the MDS step

def make_mds_values():
    section = "Instability of the MDS Step"
    text = []
    for data, method, name in DATASETS:
        r = pd.read_csv(
            CONFIG["ILLUSTRATION_DIR"] / f"mds_instability_summary_{data}_{method}.csv"
        ).iloc[0]
        text.append(
            text_row(
                section,
                f"Pairwise distance among MDS runs, {name} with m={int(r['n_runs_base'])}",
                r["mean"],
                r["sd"],
                digits=3,
            )
        )
    return text


if __name__ == "__main__":
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    make_simulation_tables()
    main_text = make_random_init_tables() + make_missing_values() + make_multiscale_tables()
    save(pd.DataFrame(main_text), "main_text_values.csv")
    save(pd.DataFrame(make_mds_values()), "supp_text_values.csv")
