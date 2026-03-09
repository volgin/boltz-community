"""Benchmark dataset preparation for affinity evaluation.

Provides utilities to prepare benchmark datasets from public sources.
Each dataset function returns a DataFrame with standardized columns:
  - id: unique compound/pair identifier
  - target_id: protein target identifier
  - experimental: experimental affinity value (log10 scale)
  - smiles: ligand SMILES (where available)

Datasets:
  - FEP+ public benchmark (Schrodinger protein-ligand-benchmark)
  - CASP16 affinity targets (public experimental data)

Note: These datasets require manual download of source data.
The functions here handle parsing and standardization.
"""

from pathlib import Path

import pandas as pd


def prepare_fep_benchmark(
    plb_dir: str | Path,
    targets: list[str] | None = None,
    affinity_types: tuple[str, ...] = ("IC50", "Ki", "Kd"),
) -> pd.DataFrame:
    """Prepare FEP+ benchmark from Schrodinger protein-ligand-benchmark.

    Downloads / parses the public protein-ligand-benchmark dataset.
    See: https://github.com/schrodinger/public_binding_free_energy_benchmark

    Parameters
    ----------
    plb_dir : path
        Path to cloned protein-ligand-benchmark repository.
    targets : list of str, optional
        Specific targets to include. If None, includes all.
        The Boltz-2 paper uses: CDK2, TYK2, JNK1, p38 (87 neutral compounds).
    affinity_types : tuple of str
        Affinity measurement types to include.

    Returns
    -------
    DataFrame with columns: id, target_id, experimental, smiles, affinity_type.
    """
    plb_dir = Path(plb_dir)

    records = []
    targets_dir = plb_dir / "data"
    if not targets_dir.exists():
        msg = (
            f"protein-ligand-benchmark data not found at {targets_dir}. "
            "Clone from: https://github.com/schrodinger/public_binding_free_energy_benchmark"
        )
        raise FileNotFoundError(msg)

    for target_dir in sorted(targets_dir.iterdir()):
        if not target_dir.is_dir():
            continue
        target_name = target_dir.name
        if targets and target_name not in targets:
            continue

        # Look for ligands CSV or SDF with affinity data
        ligands_csv = target_dir / "ligands.csv"
        if ligands_csv.exists():
            df = pd.read_csv(ligands_csv)
            for _, row in df.iterrows():
                record = {
                    "id": f"{target_name}_{row.get('name', row.get('ID', ''))}",
                    "target_id": target_name,
                    "smiles": row.get("smiles", row.get("SMILES", "")),
                }
                # Find affinity value
                for aff_type in affinity_types:
                    col_candidates = [
                        aff_type, f"exp_{aff_type}", f"experimental_{aff_type}",
                        f"pIC50", f"p{aff_type}",
                    ]
                    for col in col_candidates:
                        if col in row and pd.notna(row[col]):
                            record["experimental"] = float(row[col])
                            record["affinity_type"] = aff_type
                            break
                    if "experimental" in record:
                        break

                if "experimental" in record:
                    records.append(record)

    if not records:
        msg = f"No affinity data found in {plb_dir}"
        raise ValueError(msg)

    return pd.DataFrame(records)


def prepare_casp16_affinity(
    casp16_dir: str | Path,
) -> pd.DataFrame:
    """Prepare CASP16 affinity benchmark from public data.

    CASP16 affinity targets include:
    - L1000: 17 protein-ligand pairs (Stage 1)
    - L3000: 93 + 30 pairs from pharma targets (Stages 2+3)

    The experimental data is available from the CASP website after the
    competition closes.

    Parameters
    ----------
    casp16_dir : path
        Directory containing CASP16 affinity data files.
        Expected files: targets.csv with columns [target_id, ligand_id,
        smiles, experimental_pIC50 or experimental_logKd].

    Returns
    -------
    DataFrame with columns: id, target_id, experimental, smiles.
    """
    casp16_dir = Path(casp16_dir)

    # Try to find the data file
    candidates = [
        casp16_dir / "targets.csv",
        casp16_dir / "affinity_targets.csv",
        casp16_dir / "casp16_affinity.csv",
    ]
    data_file = None
    for c in candidates:
        if c.exists():
            data_file = c
            break

    if data_file is None:
        msg = (
            f"CASP16 affinity data not found in {casp16_dir}. "
            "Expected a CSV file with columns: target_id, ligand_id, smiles, "
            "and experimental affinity values. "
            "Download from: https://predictioncenter.org/casp16/"
        )
        raise FileNotFoundError(msg)

    df = pd.read_csv(data_file)

    result = pd.DataFrame()
    result["id"] = df.apply(
        lambda r: f"{r.get('target_id', '')}_{r.get('ligand_id', r.name)}", axis=1
    )
    result["target_id"] = df.get("target_id", df.iloc[:, 0])

    # Find experimental column
    exp_cols = [c for c in df.columns if "exp" in c.lower() or "affinity" in c.lower()
                or "pic50" in c.lower() or "logk" in c.lower() or "ic50" in c.lower()]
    if exp_cols:
        result["experimental"] = df[exp_cols[0]].astype(float)
    else:
        msg = f"No experimental affinity column found. Columns: {list(df.columns)}"
        raise ValueError(msg)

    if "smiles" in df.columns or "SMILES" in df.columns:
        result["smiles"] = df.get("smiles", df.get("SMILES"))

    return result


def load_ground_truth(
    path: str | Path,
    id_col: str = "id",
    exp_col: str = "experimental",
    target_col: str | None = "target_id",
    active_col: str | None = None,
) -> pd.DataFrame:
    """Load ground truth from a generic CSV file.

    Parameters
    ----------
    path : str or Path
        Path to CSV file.
    id_col : str
        Column name for compound/pair identifiers.
    exp_col : str
        Column name for experimental values.
    target_col : str, optional
        Column name for target identifiers.
    active_col : str, optional
        Column name for binary activity labels.

    Returns
    -------
    Standardized DataFrame.
    """
    df = pd.read_csv(path)
    result = pd.DataFrame()
    result["id"] = df[id_col]
    result["experimental"] = df[exp_col].astype(float)
    if target_col and target_col in df.columns:
        result["target_id"] = df[target_col]
    if active_col and active_col in df.columns:
        result["is_active"] = df[active_col].astype(bool)
    return result
