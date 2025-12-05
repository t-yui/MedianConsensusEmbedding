# MedianConsensusEmbedding

This repository is dedicated to the code availability for the paper: [**Median Consensus Embedding for Dimensionality Reduction**](
https://doi.org/10.48550/arXiv.2503.08103).

## Description

Median Concensus Embedding (MCE) is an algorithm to obtain consensus representation of multiple embeddings via computing geometric median defined on embedding space.
MCE can be used to (i) obtain stable visualization from multi-run with random initialization of non-liinear dimensionality reduction methods, such as t-SNE or UMAP, (ii) integrate multiple embeddings from multiply imputed data to address missing values, and (iii) integrate embeddings generated with different hyperparameters into a single multiscale consensus representation.

## Requirements

- Python ≥ 3.9.0
- Required libraries for MCE implementation: `scikit-lean`
- Required libraries for evaluation experiments: `scikit-lean`, `umap-learn`, `tqdm`

## License

This repository is licensed under the [MIT License](LICENSE).

## Usage

### Perform MCE

```python
y, x = MCE(
    base_embeddings,
    n_components=2,
    rs_mds=1
)
```

**Arguments**

`base_embeddings` (list): List of base embeddings (numpy array, shape = ($n_{\mathrm{samples}}$, $n_{\mathrm{features}}$) with typically $n_{\mathrm{features}}$=2$) to be integrated

`n_components` (int, default $=2$): The dimension of projected eucledian space

`rs_mds` (int, default $=1$): Random state of MDS initialization

**Value**

`y` (numpy array, shape = ($n_{\mathrm{samples}}$, $n_{\mathrm{features}}$)): The consensus embedding

`x` (numpy array, shape = ($n_{\mathrm{samples}}$, $n_{\mathrm{samples}}$)): The pairwise distance matrix of consensus embedding

### Perform evaluation experiments (Coming soon)

Set the configures of each script, and execute the following commands in your shell environment.


1. Evaluation of stability of MCE

```sh
python3 experiments_random_init.py
```

2. Evaluation of instability arising from MDS

```sh
python3 experiments_mds_instability.py
```

3. Illustration of combined approach with multiple imputation and obtaining multiscale embeddings.

```sh
python3 experiments_missing_multiscale.py
```

## Citation

Tomo, Y. & Yoneoka, D. (2025). Median Consensus Embedding for Dimensionality Reduction. *arXiv preprint arXiv:2503.08103*. 

