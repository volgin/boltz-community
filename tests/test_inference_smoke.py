"""Smoke test: verify boltz predict runs and produces output.

This test runs a minimal prediction to validate the pipeline works
end-to-end. It requires the model checkpoint to be available
(downloaded on first run to ~/.boltz).

Marked as slow since it downloads the model and runs inference.
Run with: pytest tests/test_inference_smoke.py -m slow
"""

import os
import subprocess
import tempfile

import pytest

try:
    import gemmi
except ImportError:
    gemmi = None


@pytest.mark.slow
def test_predict_small_peptide():
    """Run boltz predict on a small peptide and verify output."""
    input_yaml = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: ACDEFGHIKL
      msa: empty
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write input
        input_path = os.path.join(tmpdir, "test_input.yaml")
        with open(input_path, "w") as f:
            f.write(input_yaml)

        output_dir = os.path.join(tmpdir, "output")

        # Run prediction with minimal settings
        result = subprocess.run(
            [
                "boltz", "predict", input_path,
                "--out_dir", output_dir,
                "--recycling_steps", "1",
                "--diffusion_samples", "1",
                "--no_write_full_pae",
                "--no_write_full_pde",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        assert result.returncode == 0, (
            f"boltz predict failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Find output structure file
        pred_dir = os.path.join(output_dir, "boltz_results_test_input", "predictions")
        assert os.path.isdir(pred_dir), f"Predictions directory not found: {pred_dir}"

        # Should have at least one prediction subdirectory
        pred_subdirs = os.listdir(pred_dir)
        assert len(pred_subdirs) > 0, "No prediction subdirectories found"

        # Find .cif output file
        cif_files = []
        for subdir in pred_subdirs:
            subdir_path = os.path.join(pred_dir, subdir)
            if os.path.isdir(subdir_path):
                for f in os.listdir(subdir_path):
                    if f.endswith(".cif"):
                        cif_files.append(os.path.join(subdir_path, f))

        assert len(cif_files) > 0, "No .cif output files found"

        # Verify CIF file is parseable
        cif_path = cif_files[0]
        assert os.path.getsize(cif_path) > 0, "CIF file is empty"

        if gemmi is not None:
            doc = gemmi.cif.read(cif_path)
            assert len(doc) > 0, "CIF document has no data blocks"

        # Check confidence scores exist
        confidence_dir = os.path.join(
            output_dir, "boltz_results_test_input", "predictions"
        )
        # Confidence JSON should be in the prediction subdirectory
        for subdir in pred_subdirs:
            subdir_path = os.path.join(pred_dir, subdir)
            if os.path.isdir(subdir_path):
                files = os.listdir(subdir_path)
                json_files = [f for f in files if f.endswith(".json")]
                assert len(json_files) > 0, (
                    f"No confidence JSON files in {subdir_path}"
                )
