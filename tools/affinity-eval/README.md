# affinity-eval

Model-agnostic binding affinity prediction evaluation tool.

Designed for evaluating and comparing affinity prediction models (Boltz-2, custom models, etc.) against experimental binding data.

## Installation

```bash
pip install -e tools/affinity-eval/

# With plotting support
pip install -e "tools/affinity-eval/[plot]"
```

## Quick start

### Evaluate Boltz-2 predictions

```bash
# Run Boltz-2 with affinity prediction
boltz predict inputs/ --out_dir results/

# Evaluate against ground truth
affinity-eval boltz results/boltz_results_inputs/predictions/ \
    --ground-truth labels.csv
```

Ground truth CSV format:
```csv
id,experimental,target_id
my_complex_1,-6.5,CDK2
my_complex_2,-7.2,CDK2
my_complex_3,-5.1,TYK2
```

### Evaluate any model from CSV

```bash
# Single model
affinity-eval csv predictions.csv \
    --id-col compound_id \
    --pred-col predicted_pIC50 \
    --exp-col experimental_pIC50 \
    --target-col protein

# Compare two models
affinity-eval compare boltz2_results.csv our_model_results.csv \
    --names "Boltz-2" "Our Model" \
    --pred-col predicted \
    --exp-col experimental \
    --target-col target_id
```

### Python API

```python
import numpy as np
from affinity_eval import evaluate

result = evaluate(
    predicted=np.array([-6.5, -7.2, -5.1, -8.0]),
    experimental=np.array([-6.3, -7.5, -4.8, -7.9]),
    target_ids=np.array(["CDK2", "CDK2", "TYK2", "TYK2"]),
)

print(result.summary())
print(f"Pearson R: {result.pearson_r:.4f}")
print(f"Per-target R: {result.per_target_pearson_r:.4f}")
```

## Metrics

| Metric | Description | Used in |
|--------|-------------|---------|
| **Pearson R** | Linear correlation (global) | General |
| **Per-target Pearson R** | Pearson R averaged across targets | Boltz-2 paper |
| **Spearman rho** | Rank correlation | General |
| **RMSE** | Root mean squared error | General |
| **MAE** | Mean absolute error | General |
| **EF@1%** | Enrichment factor at top 1% | Virtual screening |
| **EF@5%** | Enrichment factor at top 5% | Virtual screening |

All metrics include optional bootstrap 95% confidence intervals.

## Benchmark datasets

### FEP+ public benchmark

Uses the [Schrodinger protein-ligand-benchmark](https://github.com/schrodinger/public_binding_free_energy_benchmark):

```bash
git clone https://github.com/schrodinger/public_binding_free_energy_benchmark.git

# Prepare the 4-target subset used in Boltz-2 paper
python -c "
from affinity_eval.datasets import prepare_fep_benchmark
df = prepare_fep_benchmark(
    'public_binding_free_energy_benchmark',
    targets=['cdk2', 'tyk2', 'jnk1', 'p38'],
)
df.to_csv('fep_ground_truth.csv', index=False)
print(f'{len(df)} compounds across {df.target_id.nunique()} targets')
"
```

### CASP16 affinity

Download experimental data from [predictioncenter.org/casp16](https://predictioncenter.org/casp16/):

```bash
# Prepare after downloading
python -c "
from affinity_eval.datasets import prepare_casp16_affinity
df = prepare_casp16_affinity('casp16_data/')
df.to_csv('casp16_ground_truth.csv', index=False)
"
```

## Boltz-2 affinity output fields

Boltz-2 produces several affinity-related fields. Use `--value-field` to select:

| Field | Description | Use case |
|-------|-------------|----------|
| `affinity_pred_value` | Ensemble log10(IC50) in uM | Lead optimization (default) |
| `affinity_probability_binary` | Ensemble P(binder) | Hit discovery / screening |
| `affinity_pred_value1` | Model 1 log10(IC50) | Debugging |
| `affinity_pred_value2` | Model 2 log10(IC50) | Debugging |

## Workflow: establishing a baseline

```bash
# 1. Prepare ground truth
python -c "
from affinity_eval.datasets import prepare_fep_benchmark
df = prepare_fep_benchmark('plb/', targets=['cdk2', 'tyk2', 'jnk1', 'p38'])
df.to_csv('fep_gt.csv', index=False)
"

# 2. Generate Boltz inputs from ground truth SMILES + target structures
#    (manual step: create YAML files for each complex)

# 3. Run Boltz-2 predictions
boltz predict inputs/ --out_dir baseline_results/ --use_msa_server

# 4. Evaluate
affinity-eval boltz baseline_results/boltz_results_inputs/predictions/ \
    --ground-truth fep_gt.csv \
    --output baseline_eval.json

# 5. After making model changes, re-run and compare
affinity-eval boltz new_results/boltz_results_inputs/predictions/ \
    --ground-truth fep_gt.csv \
    --output new_eval.json
```
