"""Tests for boltz.data.types — Structure, MSA, Record serialization and remove_invalid_chains."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from boltz.data.types import (
    Atom,
    Bond,
    Chain,
    Connection,
    Interface,
    MSA,
    MSADeletion,
    MSAResidue,
    MSASequence,
    Record,
    Residue,
    Structure,
    StructureInfo,
    ChainInfo,
    InterfaceInfo,
)


class TestStructureNpzRoundtrip:
    """Test Structure NPZ serialization roundtrip."""

    def test_roundtrip(self, minimal_structure, tmp_path):
        """Structure dump → load → identical arrays."""
        path = tmp_path / "struct.npz"
        minimal_structure.dump(path)

        loaded = Structure.load(path)

        assert len(loaded.atoms) == len(minimal_structure.atoms)
        assert len(loaded.residues) == len(minimal_structure.residues)
        assert len(loaded.chains) == len(minimal_structure.chains)
        np.testing.assert_array_equal(loaded.mask, minimal_structure.mask)

        # Check coordinates match
        np.testing.assert_array_almost_equal(
            loaded.atoms["coords"], minimal_structure.atoms["coords"]
        )


class TestRecordJsonRoundtrip:
    """Test Record JSON serialization roundtrip."""

    def test_roundtrip(self, tmp_path):
        """Record dump → load → identical fields."""
        record = Record(
            id="test_record",
            structure=StructureInfo(resolution=2.5, method="X-RAY"),
            chains=[
                ChainInfo(
                    chain_id=0,
                    chain_name="A",
                    mol_type=0,
                    cluster_id="cluster1",
                    msa_id="msa1",
                    num_residues=100,
                )
            ],
            interfaces=[InterfaceInfo(chain_1=0, chain_2=1)],
        )

        path = tmp_path / "record.json"
        record.dump(path)
        loaded = Record.load(path)

        assert loaded.id == "test_record"
        assert loaded.structure.resolution == 2.5
        assert loaded.structure.method == "X-RAY"
        assert len(loaded.chains) == 1
        assert loaded.chains[0].chain_name == "A"


class TestRemoveInvalidChains:
    """Test Structure.remove_invalid_chains."""

    def test_remove_masked_chain(self, minimal_structure_with_ligand):
        """Masked chains are removed and indices re-numbered."""
        # Mask out the ligand chain (index 1)
        mask = np.array([True, False], dtype=bool)
        struct = Structure(
            atoms=minimal_structure_with_ligand.atoms,
            bonds=minimal_structure_with_ligand.bonds,
            residues=minimal_structure_with_ligand.residues,
            chains=minimal_structure_with_ligand.chains,
            connections=minimal_structure_with_ligand.connections,
            interfaces=minimal_structure_with_ligand.interfaces,
            mask=mask,
        )

        cleaned = struct.remove_invalid_chains()

        assert len(cleaned.chains) == 1
        assert len(cleaned.residues) == 3  # Only protein residues
        assert cleaned.chains[0]["asym_id"] == 0
        assert cleaned.mask.all()


class TestMSADtypes:
    """Test MSA structured array dtypes."""

    def test_msa_creation(self):
        """MSA can be created with correct dtypes."""
        sequences = np.array([(0, -1, 0, 5, 0, 0)], dtype=MSASequence)
        deletions = np.array([], dtype=MSADeletion)
        residues = np.array([(1,), (2,), (3,), (4,), (5,)], dtype=MSAResidue)

        msa = MSA(sequences=sequences, deletions=deletions, residues=residues)

        assert msa.sequences[0]["seq_idx"] == 0
        assert len(msa.residues) == 5
        assert msa.residues[2]["res_type"] == 3
