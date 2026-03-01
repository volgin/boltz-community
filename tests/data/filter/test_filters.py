"""Tests for boltz.data.filter.static — MinimumLengthFilter, UnknownFilter, ExcludedLigands, ConsecutiveCA."""

import numpy as np
import pytest

from boltz.data import const
from boltz.data.types import Atom, Bond, Chain, Connection, Interface, Residue, Structure


def _make_protein_chain(n_residues, res_type=None, coords_func=None, chain_idx=0):
    """Helper to build a protein chain with n_residues for filter testing."""
    atoms_per_res = 5
    n_atoms = n_residues * atoms_per_res

    atoms = np.zeros(n_atoms, dtype=Atom)
    for i in range(n_atoms):
        atoms[i]["name"] = [46, 0, 0, 0]  # N
        atoms[i]["element"] = 7
        atoms[i]["is_present"] = True
        if coords_func:
            atoms[i]["coords"] = coords_func(i)
        else:
            atoms[i]["coords"] = [float(i), 0.0, 0.0]

    residues = np.zeros(n_residues, dtype=Residue)
    rt = res_type if res_type is not None else const.token_ids.get("ALA", 0)
    for i in range(n_residues):
        residues[i]["name"] = "ALA"
        residues[i]["res_type"] = rt
        residues[i]["res_idx"] = i
        residues[i]["atom_idx"] = i * atoms_per_res
        residues[i]["atom_num"] = atoms_per_res
        residues[i]["atom_center"] = i * atoms_per_res + 1  # CA
        residues[i]["atom_disto"] = i * atoms_per_res + 4
        residues[i]["is_standard"] = True
        residues[i]["is_present"] = True

    chains = np.zeros(1, dtype=Chain)
    chains[0]["name"] = chr(65 + chain_idx)
    chains[0]["mol_type"] = const.chain_type_ids["PROTEIN"]
    chains[0]["entity_id"] = chain_idx
    chains[0]["asym_id"] = chain_idx
    chains[0]["atom_idx"] = 0
    chains[0]["atom_num"] = n_atoms
    chains[0]["res_idx"] = 0
    chains[0]["res_num"] = n_residues

    return atoms, residues, chains


def _make_structure(atoms, residues, chains):
    """Build a Structure from arrays."""
    return Structure(
        atoms=atoms,
        bonds=np.array([], dtype=Bond),
        residues=residues,
        chains=chains,
        connections=np.array([], dtype=Connection),
        interfaces=np.array([], dtype=Interface),
        mask=np.ones(len(chains), dtype=bool),
    )


class TestMinimumLengthFilter:
    """Tests for MinimumLengthFilter."""

    def test_filter_short_chain(self):
        """Chain with 2 residues filtered out by min_len=4."""
        from boltz.data.filter.static.polymer import MinimumLengthFilter

        atoms, residues, chains = _make_protein_chain(2)
        struct = _make_structure(atoms, residues, chains)

        f = MinimumLengthFilter(min_len=4, max_len=5000)
        mask = f.filter(struct)
        assert mask[0] == False  # noqa: E712

    def test_filter_keeps_long_chain(self):
        """Chain with 10 residues passes min_len=4."""
        from boltz.data.filter.static.polymer import MinimumLengthFilter

        atoms, residues, chains = _make_protein_chain(10)
        struct = _make_structure(atoms, residues, chains)

        f = MinimumLengthFilter(min_len=4, max_len=5000)
        mask = f.filter(struct)
        assert mask[0] == True  # noqa: E712


class TestUnknownFilter:
    """Tests for UnknownFilter."""

    def test_all_unknown_filtered(self):
        """Chain with all UNK residues is filtered."""
        from boltz.data.filter.static.polymer import UnknownFilter

        unk_id = const.unk_token_ids["PROTEIN"]
        atoms, residues, chains = _make_protein_chain(5, res_type=unk_id)
        struct = _make_structure(atoms, residues, chains)

        f = UnknownFilter()
        mask = f.filter(struct)
        assert mask[0] == False  # noqa: E712

    def test_known_residues_pass(self):
        """Chain with normal residues passes."""
        from boltz.data.filter.static.polymer import UnknownFilter

        atoms, residues, chains = _make_protein_chain(5)
        struct = _make_structure(atoms, residues, chains)

        f = UnknownFilter()
        mask = f.filter(struct)
        assert mask[0] == True  # noqa: E712


class TestExcludedLigands:
    """Tests for ExcludedLigands."""

    def test_excluded_ligand_filtered(self):
        """NONPOLYMER chain with excluded ligand name is filtered."""
        from boltz.data.filter.static.ligand import ExcludedLigands

        # Get an excluded ligand name
        excl_name = next(iter(const.ligand_exclusion))

        atoms = np.zeros(3, dtype=Atom)
        for i in range(3):
            atoms[i]["element"] = 6
            atoms[i]["is_present"] = True
            atoms[i]["coords"] = [float(i), 0.0, 0.0]

        residues = np.zeros(1, dtype=Residue)
        residues[0]["name"] = excl_name
        residues[0]["atom_idx"] = 0
        residues[0]["atom_num"] = 3
        residues[0]["is_present"] = True

        chains = np.zeros(1, dtype=Chain)
        chains[0]["name"] = "A"
        chains[0]["mol_type"] = const.chain_type_ids["NONPOLYMER"]
        chains[0]["res_idx"] = 0
        chains[0]["res_num"] = 1

        struct = _make_structure(atoms, residues, chains)
        f = ExcludedLigands()
        mask = f.filter(struct)
        assert mask[0] == False  # noqa: E712


class TestConsecutiveCA:
    """Tests for ConsecutiveCA."""

    def test_large_gap_filtered(self):
        """Chain with consecutive CA distance > max_dist is filtered."""
        from boltz.data.filter.static.polymer import ConsecutiveCA

        def coords_func(i):
            res_idx = i // 5
            # Place CA atoms (idx 1 within each residue) at distances:
            # res 0: CA at (0, 0, 0), res 1: CA at (3.8, 0, 0), res 2: CA at (50, 0, 0)
            if i % 5 == 1:  # CA
                if res_idx == 0:
                    return [0.0, 0.0, 0.0]
                elif res_idx == 1:
                    return [3.8, 0.0, 0.0]
                else:
                    return [50.0, 0.0, 0.0]
            return [float(res_idx * 5 + i % 5), 0.0, 0.0]

        atoms, residues, chains = _make_protein_chain(3, coords_func=coords_func)
        struct = _make_structure(atoms, residues, chains)

        f = ConsecutiveCA(max_dist=10.0)
        mask = f.filter(struct)
        assert mask[0] == False  # noqa: E712

    def test_normal_chain_passes(self):
        """Chain with normal CA spacing passes."""
        from boltz.data.filter.static.polymer import ConsecutiveCA

        def coords_func(i):
            res_idx = i // 5
            if i % 5 == 1:  # CA
                return [res_idx * 3.8, 0.0, 0.0]
            return [float(res_idx * 5 + i % 5), 0.0, 0.0]

        atoms, residues, chains = _make_protein_chain(5, coords_func=coords_func)
        struct = _make_structure(atoms, residues, chains)

        f = ConsecutiveCA(max_dist=10.0)
        mask = f.filter(struct)
        assert mask[0] == True  # noqa: E712
