#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

run() {
    local name="$1"
    echo "=== Running ${name} ==="
    python3 -u "scripts/${name}.py" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

# simulation study
run simulation_lowdim
run simulation_highdim

# evaluation of rapid convergence and stability
run experiments_random_init
run experiments_random_init_eb_umap

# combined approach with multiple imputation
run experiments_missing

# consensus representation of multiscale hyperparameters
run experiments_multiscale

# instability of the MDS step
run experiments_mds_instability
run experiments_mds_instability_eb_umap
