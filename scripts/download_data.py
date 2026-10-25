#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import shutil
import urllib.request
import warnings
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "toxo": {
        "url": (
            "https://raw.githubusercontent.com/lgatto/pRolocdata/"
            "f717d11e22beab97deca6b82306b00367cf55fca/data/Barylyuk2020ToxoLopit.rda"
        ),
        "sha256": "4eac2f0da2f5891ef658ef3865f4a14e9b660d81876883d9213e1fbc60c3afd2",
        "path": ROOT / "Barylyuk2020ToxoLopit.rda",
    },
    "eb": {
        "url": (
            "https://raw.githubusercontent.com/KrishnaswamyLab/PHATE/"
            "65a1f73f0c53a65bf1f8db3d4ad32dbef73595c3/data/EBdata.mat"
        ),
        "sha256": "9909b75cf0ab142710d9166989ea0ddf2fd18e6424f43229a6a6b30486cb1e70",
        "path": ROOT / "EBdata.mat",
    },
}

TOXO_DATA_CSV = ROOT / "data_Barylyuk2020ToxoLopit.csv"
TOXO_LABEL_CSV = ROOT / "label_Barylyuk2020ToxoLopit.csv"


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path, sha256):
    if path.exists() and sha256sum(path) == sha256:
        print(f"Already downloaded: {path.name}")
        return
    print(f"Downloading: {url}")
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    if sha256sum(tmp) != sha256:
        tmp.unlink()
        raise RuntimeError(f"Checksum mismatch: {path.name}")
    tmp.replace(path)
    print(f"Saved: {path}")


def convert_toxo(rda_path, data_csv, label_csv):
    """
    extract the expression profiles and the marker labels from the MSnSet object
    """
    import rdata

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        msnset = rdata.read_rda(str(rda_path))["Barylyuk2020ToxoLopit"]

    exprs = msnset.assayData["exprs"]
    data_df = pd.DataFrame(
        exprs.values,
        index=exprs.coords["dim_0"].values,
        columns=exprs.coords["dim_1"].values,
    )
    label_df = msnset.featureData.data[["markers"]]
    if not data_df.index.equals(label_df.index):
        raise RuntimeError("Protein IDs of exprs and fData do not match.")

    data_df.to_csv(data_csv)
    label_df.to_csv(label_csv)
    print(f"Saved: {data_csv} ({data_df.shape[0]} proteins x {data_df.shape[1]} fractions)")
    print(f"Saved: {label_csv} ({(label_df['markers'] != 'unknown').sum()} marker proteins)")


if __name__ == "__main__":
    for source in SOURCES.values():
        download(source["url"], source["path"], source["sha256"])
    convert_toxo(SOURCES["toxo"]["path"], TOXO_DATA_CSV, TOXO_LABEL_CSV)
