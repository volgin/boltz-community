"""Tests for boltz.data.write.pdb — to_pdb."""

import torch
import pytest

from boltz.data import const
from boltz.data.write.pdb import to_pdb


class TestToPdb:
    """Tests for to_pdb."""

    def test_atom_records(self, minimal_structure):
        """Output contains ATOM records."""
        pdb_str = to_pdb(minimal_structure)
        lines = pdb_str.split("\n")
        atom_lines = [l for l in lines if l.startswith("ATOM")]
        assert len(atom_lines) > 0

    def test_ter_and_end(self, minimal_structure_with_ligand):
        """Output contains TER between chains and END at the end."""
        pdb_str = to_pdb(minimal_structure_with_ligand)
        lines = pdb_str.split("\n")
        ter_lines = [l for l in lines if l.strip().startswith("TER")]
        end_lines = [l for l in lines if l.strip().startswith("END")]
        assert len(ter_lines) >= 1
        assert len(end_lines) >= 1

    def test_column_alignment(self, minimal_structure):
        """PDB lines are 80 chars wide."""
        pdb_str = to_pdb(minimal_structure)
        lines = pdb_str.split("\n")
        for line in lines:
            if line.strip():
                assert len(line) == 80

    def test_plddt_to_bfactor(self, minimal_structure):
        """pLDDT values appear in B-factor column (cols 61-66)."""
        # 3 residues → 3 plddt values
        plddts = torch.tensor([0.85, 0.70, 0.95])
        pdb_str = to_pdb(minimal_structure, plddts=plddts)

        lines = [l for l in pdb_str.split("\n") if l.startswith("ATOM")]
        # First residue atoms should have B-factor ≈ 85.0
        first_atom = lines[0]
        bfactor_str = first_atom[60:66].strip()
        bfactor = float(bfactor_str)
        assert bfactor == pytest.approx(85.0, abs=0.1)

    def test_hetatm_for_ligands(self, minimal_structure_with_ligand):
        """NONPOLYMER chains use HETATM records."""
        pdb_str = to_pdb(minimal_structure_with_ligand)
        lines = pdb_str.split("\n")
        hetatm_lines = [l for l in lines if l.startswith("HETATM")]
        assert len(hetatm_lines) > 0
