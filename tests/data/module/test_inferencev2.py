from pathlib import Path

import pytest

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest


def test_predict_dataloader_uses_configured_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("boltz.data.module.inferencev2.load_canonicals", lambda _: {})

    data_module = Boltz2InferenceDataModule(
        manifest=Manifest(records=[]),
        target_dir=tmp_path / "structures",
        msa_dir=tmp_path / "msa",
        mol_dir=tmp_path / "mols",
        num_workers=0,
        batch_size=3,
    )

    assert data_module.predict_dataloader().batch_size == 3


def test_predict_dataloader_defaults_to_batch_size_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("boltz.data.module.inferencev2.load_canonicals", lambda _: {})

    data_module = Boltz2InferenceDataModule(
        manifest=Manifest(records=[]),
        target_dir=tmp_path / "structures",
        msa_dir=tmp_path / "msa",
        mol_dir=tmp_path / "mols",
        num_workers=0,
    )

    assert data_module.predict_dataloader().batch_size == 1


def test_affinity_predict_dataloader_rejects_batch_size_above_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("boltz.data.module.inferencev2.load_canonicals", lambda _: {})

    with pytest.raises(ValueError, match="Affinity inference only supports batch_size=1."):
        Boltz2InferenceDataModule(
            manifest=Manifest(records=[]),
            target_dir=tmp_path / "structures",
            msa_dir=tmp_path / "msa",
            mol_dir=tmp_path / "mols",
            num_workers=0,
            batch_size=2,
            affinity=True,
        )
