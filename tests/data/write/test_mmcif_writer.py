"""Tests for boltz.data.write.mmcif — to_mmcif."""

import pytest

from boltz.data.write.mmcif import to_mmcif


class TestToMmcif:
    """Tests for to_mmcif."""

    def test_basic_output(self, minimal_structure):
        """Output contains _atom_site records."""
        mmcif_str = to_mmcif(minimal_structure)
        assert "_atom_site" in mmcif_str
        assert len(mmcif_str) > 100

    def test_gemmi_roundtrip(self, minimal_structure):
        """Output is parseable by gemmi."""
        gemmi = pytest.importorskip("gemmi")
        mmcif_str = to_mmcif(minimal_structure)

        doc = gemmi.cif.read_string(mmcif_str)
        assert len(doc) > 0
        block = doc[0]
        # Should have atom records via _atom_site category
        # gemmi uses find_loop or find for block
        loop = block.find_loop("_atom_site.id")
        assert loop is not None

    def test_entity_type(self, minimal_structure):
        """Protein entity produces valid mmCIF output."""
        mmcif_str = to_mmcif(minimal_structure)

        # Should contain entity information and atom records
        assert "_atom_site" in mmcif_str
        # Check there's meaningful content
        lines = mmcif_str.split("\n")
        assert len(lines) > 10
