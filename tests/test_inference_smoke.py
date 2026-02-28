"""Smoke tests: verify boltz predict runs and produces output.

These tests run minimal predictions to validate the pipeline works
end-to-end. They require the model checkpoint to be available
(downloaded on first run to ~/.boltz).

Marked as slow since they download the model and run inference.
Run with: pytest tests/test_inference_smoke.py -m slow
"""

import json
import os
import subprocess
import tempfile

import pytest

try:
    import gemmi
except ImportError:
    gemmi = None


def _run_boltz_predict(input_yaml, input_filename, tmpdir, extra_args=None):
    """Run boltz predict and return (result, predictions_dir)."""
    input_path = os.path.join(tmpdir, input_filename)
    with open(input_path, "w") as f:
        f.write(input_yaml)

    output_dir = os.path.join(tmpdir, "output")
    cmd = [
        "boltz", "predict", input_path,
        "--out_dir", output_dir,
        "--recycling_steps", "1",
        "--diffusion_samples", "1",
        "--no_write_full_pae",
        "--no_write_full_pde",
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    stem = os.path.splitext(input_filename)[0]
    pred_dir = os.path.join(output_dir, f"boltz_results_{stem}", "predictions")
    return result, pred_dir


def _find_files(pred_dir, extension):
    """Find all files with given extension in prediction subdirectories."""
    found = []
    if not os.path.isdir(pred_dir):
        return found
    for subdir in os.listdir(pred_dir):
        subdir_path = os.path.join(pred_dir, subdir)
        if os.path.isdir(subdir_path):
            for f in os.listdir(subdir_path):
                if f.endswith(extension):
                    found.append(os.path.join(subdir_path, f))
    return found


@pytest.mark.slow
def test_predict_small_peptide():
    """Run boltz predict on a small peptide and verify structure output."""
    input_yaml = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: ACDEFGHIKL
      msa: empty
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, pred_dir = _run_boltz_predict(
            input_yaml, "test_input.yaml", tmpdir,
        )

        assert result.returncode == 0, (
            f"boltz predict failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        assert os.path.isdir(pred_dir), f"Predictions directory not found: {pred_dir}"

        # Verify CIF structure output
        cif_files = _find_files(pred_dir, ".cif")
        assert len(cif_files) > 0, "No .cif output files found"
        assert os.path.getsize(cif_files[0]) > 0, "CIF file is empty"

        if gemmi is not None:
            doc = gemmi.cif.read(cif_files[0])
            assert len(doc) > 0, "CIF document has no data blocks"

        # Verify confidence JSON output
        json_files = _find_files(pred_dir, ".json")
        assert len(json_files) > 0, "No confidence JSON files found"


@pytest.mark.slow
def test_predict_affinity():
    """Run boltz predict with protein+ligand affinity and verify output.

    This mirrors the Phyra Boltz worker's critical path:
    protein + SMILES ligand with affinity prediction requested.
    Validates that affinity_probability_binary and affinity_pred_value
    are present, numeric, and in expected ranges.
    """
    # Tyrosine (small, drug-like) against a short peptide.
    # Small enough for fast inference, complex enough to exercise
    # the affinity head with a real ligand.
    input_yaml = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: ACDEFGHIKL
      msa: empty
  - ligand:
      id: L1
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
properties:
  - affinity:
      binder: L1
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, pred_dir = _run_boltz_predict(
            input_yaml, "test_affinity.yaml", tmpdir,
        )

        assert result.returncode == 0, (
            f"boltz predict failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        assert os.path.isdir(pred_dir), f"Predictions directory not found: {pred_dir}"

        # --- Affinity JSON ---
        affinity_files = _find_files(pred_dir, ".json")
        affinity_file = None
        for f in affinity_files:
            if os.path.basename(f).startswith("affinity_"):
                affinity_file = f
                break

        assert affinity_file is not None, (
            f"No affinity_*.json file found. Files present: "
            f"{[os.path.basename(f) for f in affinity_files]}"
        )

        with open(affinity_file) as f:
            affinity_data = json.load(f)

        # affinity_probability_binary: probability of being a binder (0 to 1)
        prob = affinity_data.get("affinity_probability_binary")
        assert prob is not None, (
            f"affinity_probability_binary missing from {affinity_file}. "
            f"Keys present: {list(affinity_data.keys())}"
        )
        assert isinstance(prob, (int, float)), (
            f"affinity_probability_binary is not numeric: {prob!r}"
        )
        assert 0.0 <= prob <= 1.0, (
            f"affinity_probability_binary out of range [0, 1]: {prob}"
        )

        # affinity_pred_value: log10(IC50 in uM), typically -3 to +4
        pred_val = affinity_data.get("affinity_pred_value")
        assert pred_val is not None, (
            f"affinity_pred_value missing from {affinity_file}. "
            f"Keys present: {list(affinity_data.keys())}"
        )
        assert isinstance(pred_val, (int, float)), (
            f"affinity_pred_value is not numeric: {pred_val!r}"
        )
        # Generous range: strong binder (-6) to non-binder (+6)
        assert -10.0 <= pred_val <= 10.0, (
            f"affinity_pred_value outside plausible range [-10, 10]: {pred_val}"
        )

        # --- Structure CIF with ligand ---
        cif_files = _find_files(pred_dir, ".cif")
        assert len(cif_files) > 0, "No .cif output files found"

        if gemmi is not None:
            structure = gemmi.read_structure(cif_files[0])
            # Verify ligand chain L1 exists in the output
            ligand_chain = None
            for model in structure:
                for chain in model:
                    if chain.name == "L1":
                        ligand_chain = chain
                        break
            assert ligand_chain is not None, (
                f"Ligand chain L1 not found in CIF. "
                f"Chains present: {[c.name for m in structure for c in m]}"
            )
            # Ligand should have atoms
            atom_count = sum(
                len(residue) for residue in ligand_chain
            )
            assert atom_count > 0, "Ligand chain L1 has no atoms"
