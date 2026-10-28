# MedianConsensusEmbedding

This repository is dedicated to the code availability for the paper: [**Median Consensus Embedding for Dimensionality Reduction**](
https://doi.org/10.48550/arXiv.2503.08103).
The GitHub repository of this project is available at [**https://github.com/t-yui/MedianConsensusEmbedding**](
https://github.com/t-yui/MedianConsensusEmbedding).

## Description

Median Consensus Embedding (MCE) is an algorithm to obtain consensus representation of multiple embeddings via computing geometric median defined on embedding space.
MCE can be used to (i) obtain stable visualization from multi-run with random initialization of non-linear dimensionality reduction methods, such as t-SNE or UMAP, (ii) integrate multiple embeddings from multiply imputed data to address missing values, and (iii) integrate embeddings generated with different hyperparameters into a single multiscale consensus representation.

## Requirements

- Python ≥ 3.9.0
- Required libraries for MCE implementation: `scikit-learn`
- Required libraries for evaluation experiments: `scikit-learn`, `umap-learn`, `tqdm`

## License

This repository is licensed under the [MIT License](LICENSE).

## Usage

### Perform MCE

```python
from median_consensus_embedding import MCE

y, x = MCE(
    base_embeddings,
    n_components=2,
    rs_mds=1
)
```

**Arguments**

`base_embeddings` (list): List of base embeddings (numpy array, shape = ($n_{\mathrm{samples}}$, $n_{\mathrm{features}}$) with typically $n_{\mathrm{features}}=2$) to be integrated

`n_components` (int, default $=2$): The dimension of projected Euclidean space

`rs_mds` (int, default $=1$): Random state of MDS initialization

**Value**

`y` (numpy array, shape = ($n_{\mathrm{samples}}$, $n_{\mathrm{features}}$)): The consensus embedding

`x` (numpy array, shape = ($n_{\mathrm{samples}}$, $n_{\mathrm{samples}}$)): The pairwise distance matrix of consensus embedding

### Perform evaluation experiments

For preparation, execute the following commands.

```
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

To perform all experiments, execute the following command.

```sh
bash run_all_experiments.sh
```

To perform each experiment separately, set the configures of each script and datasets and execute the following commands.

1. Simulation experiment (low-dimensional scenarios)

```sh
python3 simulation_lowdim.py
```

2. Simulation experiment (high-dimensional scenarios)

```sh
python3 simulation_highdim.py
```

3. Downloading data before illustration experiments

```sh
python3 download_data.py
```

4. Evaluation of stability[^1]

```sh
python3 experiments_random_init.py
```

5. Evaluation of instability arising from MDS

```sh
python3 experiments_mds_instability.py
```

6. Illustration of combined approach with multiple imputation

```sh
python3 experiments_missing.py
```

7. Illustration of integrating multi-scale embeddings

```sh
python3 experiments_multiscale.py
```

8. Creating tables and abstracting values in the paper

```sh
python3 make_tables.py
```

[^1]: Specify the dataset (`toxo` or `eb`) and the method (`tsne` or `umap`) in the config. The paper uses `toxo` with `tsne` and `eb` with `umap`.

## Citation

Tomo, Y. & Yoneoka, D. (2025). Median Consensus Embedding for Dimensionality Reduction. *arXiv preprint arXiv:2503.08103*. 
