"""Tests for boltz.data.parse.mmcif and mmcif_with_constraints — CIF parsing."""

from pathlib import Path

import pytest
from rdkit import Chem


class TestEmptyCifParsing:
    """Both mmCIF parsers must reject CIF files that contain no coordinate models."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        """Skip if gemmi or parser deps are missing."""
        try:
            import gemmi  # noqa: F401
            from boltz.data.parse.mmcif import parse_mmcif  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Cannot import parser deps: {e}")

    def _empty_cif(self, tmp_path, name="empty.cif"):
        """Write a minimal CIF with no coordinate models."""
        cif_content = """\
data_EMPTY
_entry.id EMPTY
_entity.id 1
_entity.type polymer
_entity.pdbx_description 'Empty test'
"""
        cif_path = tmp_path / name
        cif_path.write_text(cif_content)
        return cif_path

    def test_parse_mmcif_raises_valueerror(self, tmp_path):
        """CIF with no coordinate models raises ValueError, not IndexError."""
        from boltz.data.parse.mmcif import parse_mmcif

        cif_path = self._empty_cif(tmp_path)

        with pytest.raises(ValueError, match="no models"):
            parse_mmcif(str(cif_path))

    def test_parse_mmcif_with_constraints_raises_valueerror(self, tmp_path):
        """mmcif_with_constraints parser also guards against empty CIF."""
        try:
            from boltz.data.parse.mmcif_with_constraints import parse_mmcif as parse_mmcif_wc
        except ImportError as e:
            pytest.skip(f"Cannot import mmcif_with_constraints: {e}")

        cif_path = self._empty_cif(tmp_path, name="empty_wc.cif")

        with pytest.raises(ValueError, match="no models"):
            parse_mmcif_wc(str(cif_path))


class TestPdbTemplateParsing:
    """PDB template parsing should tolerate missing entity sequences."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        """Skip if parser deps are missing."""
        try:
            import gemmi  # noqa: F401
            from boltz.data.parse.pdb import parse_pdb  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Cannot import parser deps: {e}")

    @staticmethod
    def _write_atom_only_pdb(tmp_path: Path, name: str = "template.pdb") -> Path:
        """Write a minimal atom-only PDB with no SEQRES metadata."""
        pdb_content = """\
ATOM      1  N   ALA A   1      11.104  13.207  14.101  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207  14.101  1.00 20.00           C
ATOM      3  C   ALA A   1      13.000  14.500  14.800  1.00 20.00           C
ATOM      4  O   ALA A   1      12.300  15.500  14.700  1.00 20.00           O
ATOM      5  CB  ALA A   1      13.100  12.000  14.900  1.00 20.00           C
TER
END
"""
        pdb_path = tmp_path / name
        pdb_path.write_text(pdb_content)
        return pdb_path

    @staticmethod
    def _mock_residue_mol(res_name, _mols, _moldir):
        """Build a small reference residue mol with atom-name properties."""
        from boltz.data import const

        mol = Chem.RWMol()
        coords = []
        for atom_name in const.ref_atoms[res_name]:
            if atom_name.startswith("N"):
                atomic_num = 7
            elif atom_name.startswith("O"):
                atomic_num = 8
            elif atom_name.startswith("S"):
                atomic_num = 16
            elif atom_name.startswith("P"):
                atomic_num = 15
            else:
                atomic_num = 6
            idx = mol.AddAtom(Chem.Atom(atomic_num))
            mol.GetAtomWithIdx(idx).SetProp("name", atom_name)
            coords.append((float(idx), 0.0, 0.0))
            if idx > 0:
                mol.AddBond(idx - 1, idx, Chem.BondType.SINGLE)

        mol = mol.GetMol()
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for idx, (x, y, z) in enumerate(coords):
            conformer.SetAtomPosition(idx, (x, y, z))
        mol.AddConformer(conformer)
        return mol

    def test_parse_pdb_falls_back_to_polymer_residues(self, tmp_path, monkeypatch):
        """Atom-only template PDBs should parse instead of indexing past sequence."""
        from boltz.data.parse.pdb import parse_pdb

        pdb_path = self._write_atom_only_pdb(tmp_path)
        monkeypatch.setattr(
            "boltz.data.parse.mmcif.get_mol",
            self._mock_residue_mol,
        )

        parsed = parse_pdb(
            str(pdb_path),
            mols={},
            moldir=str(tmp_path),
            use_assembly=False,
            compute_interfaces=False,
        )

        assert parsed.sequences == {"A1": "A"}
        assert len(parsed.data.chains) == 1
        assert len(parsed.data.residues) == 1
