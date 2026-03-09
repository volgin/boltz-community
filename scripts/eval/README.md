# Boltz Structural & Affinity Evaluation

This directory contains scripts for evaluating Boltz model predictions against reference structures and experimental binding affinities.

## Overview

| Tier | What | Scripts | Requirements |
|------|------|---------|--------------|
| **1** | Boltz-1 structural benchmarks | `run_boltz_eval.py`, `aggregate_boltz_eval.py` | OpenStructure Docker, GPU |
| **2** | Boltz-2 structural benchmarks | Same scripts | OpenStructure Docker, GPU |
| **3** | Affinity prediction benchmarks | `tools/affinity-eval/` (standalone) | Python only |

## Tier 1 & 2: Structural Evaluation

### Prerequisites

1. **OpenStructure v2.8.0** Docker image (required for reproducing published metrics):
   ```bash
   docker pull registry.scicore.unibas.ch/schwede/openstructure:2.8.0
   docker tag registry.scicore.unibas.ch/schwede/openstructure:2.8.0 openstructure-0.2.8
   ```

2. **Benchmark data** — download from [Google Drive](https://drive.google.com/file/d/1JvHlYUMINOaqPTunI9wBYrfYniKgVmxf/view?usp=sharing) (Boltz-1 eval data with targets, inputs, reference outputs):
   ```bash
   # Unzip into a working directory
   unzip boltz_results_final.zip -d /path/to/eval_data
   ```

### Step 1: Run predictions

Run Boltz on the benchmark targets:

```bash
# Boltz-1
boltz predict /path/to/eval_data/inputs/test/boltz/ \
    --out_dir /path/to/my_results/test/ \
    --recycling_steps 10 \
    --sampling_steps 200 \
    --diffusion_samples 5 \
    --use_msa_server \
    --model boltz1

# Boltz-2
boltz predict /path/to/eval_data/inputs/test/boltz/ \
    --out_dir /path/to/my_results_v2/test/ \
    --recycling_steps 10 \
    --sampling_steps 200 \
    --diffusion_samples 5 \
    --use_msa_server
```

### Step 2: Run OpenStructure evaluation

```bash
python scripts/eval/run_boltz_eval.py \
    /path/to/my_results/test/predictions/ \
    /path/to/eval_data/targets/test/ \
    /path/to/my_evals/test/ \
    --mount /path/to  \
    --num-samples 5
```

### Step 3: Aggregate results

```bash
python scripts/eval/aggregate_boltz_eval.py \
    /path/to/my_results/test/predictions/ \
    /path/to/my_evals/test/ \
    --output results.csv

# Compare against a baseline
python scripts/eval/aggregate_boltz_eval.py \
    /path/to/my_results/test/predictions/ \
    /path/to/my_evals/test/ \
    --baseline /path/to/baseline_results.csv \
    --output results.csv
```

### Metrics computed

**Polymer metrics** (via OpenStructure `compare-structures`):
- lDDT (Local Distance Difference Test)
- bb-lDDT (backbone lDDT)
- TM-score
- DockQ (>0.23 and >0.49 thresholds)
- QS-score, ICS, IPS, rigid scores, patch scores

**Ligand metrics** (via OpenStructure `compare-ligand-structures`):
- lDDT-PLI (protein-ligand interface lDDT)
- L-RMSD (<2A, <5A thresholds)

### Matching published results

The Boltz-1 paper used:
- 541 PDB test targets (40% seq similarity, 80% Tanimoto cutoff)
- 66 CASP15 targets
- 10 recycling steps, 200 sampling steps, 5 samples
- Oracle (best of 5) and Top-1 (best by confidence) selection
- OpenStructure v2.8.0

## Tier 3: Affinity Evaluation

See [`tools/affinity-eval/`](../../tools/affinity-eval/) for a standalone, model-agnostic affinity evaluation tool.

```bash
pip install -e tools/affinity-eval/

# Evaluate Boltz predictions
affinity-eval boltz /path/to/boltz_predictions/ --ground-truth labels.csv

# Evaluate any model from a CSV
affinity-eval csv predictions.csv --id-col target --pred-col affinity --exp-col experimental

# Compare two models
affinity-eval compare model_a.csv model_b.csv --names "Boltz-2" "Our Model"
```

## Legacy scripts

The original upstream evaluation scripts are preserved for reference:
- `run_evals.py` — original multi-model OpenStructure runner (Boltz-1 era)
- `aggregate_evals.py` — original multi-model aggregation with plots
- `physcialsim_metrics.py` — physical validity checks (hardcoded paths)
