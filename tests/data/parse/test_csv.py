"""Tests for boltz.data.parse.csv — parse_csv."""

from pathlib import Path

import pytest

from boltz.data.parse.csv import parse_csv


@pytest.fixture
def basic_csv(tmp_path):
    """Create a basic CSV file with sequence and key columns."""
    content = "key,sequence\n123,ACDEF\n456,GHIKL\n789,MNPQR\n"
    p = tmp_path / "test.csv"
    p.write_text(content)
    return p


@pytest.fixture
def invalid_csv(tmp_path):
    """Create a CSV with wrong column names."""
    content = "name,seq\nfoo,ACDEF\n"
    p = tmp_path / "test_invalid.csv"
    p.write_text(content)
    return p


@pytest.fixture
def duplicate_csv(tmp_path):
    """Create a CSV with duplicate sequences, some paired some unpaired."""
    content = "key,sequence\n123,ACDEF\nnan,ACDEF\n456,ACDEF\nnan,GHIKL\nnan,GHIKL\n"
    p = tmp_path / "test_dup.csv"
    p.write_text(content)
    return p


class TestParseCsv:
    """Tests for parse_csv."""

    def test_basic(self, basic_csv):
        """Synthetic CSV → valid MSA structure."""
        msa = parse_csv(basic_csv, max_seqs=None)

        assert len(msa.sequences) == 3
        assert len(msa.residues) > 0
        # Each sequence has 5 residues
        for seq in msa.sequences:
            length = seq["res_end"] - seq["res_start"]
            assert length == 5

    def test_invalid_columns(self, invalid_csv):
        """Wrong column names → ValueError."""
        with pytest.raises(ValueError, match="Invalid CSV format"):
            parse_csv(invalid_csv)

    def test_taxonomy_key(self, basic_csv):
        """Non-NaN key → taxonomy in MSASequence."""
        msa = parse_csv(basic_csv, max_seqs=None)

        # All keys are numeric strings, should be stored as taxonomy_id
        for seq in msa.sequences:
            assert seq["taxonomy"] != -1

    def test_duplicate_handling(self, duplicate_csv):
        """Unpaired duplicates skipped, paired kept."""
        msa = parse_csv(duplicate_csv, max_seqs=None)

        # "ACDEF" appears 3 times: keys 123, nan, 456
        # First occurrence (key=123) is kept
        # Second (nan) is unpaired duplicate → skipped
        # Third (key=456) is paired → kept
        # "GHIKL" appears twice with nan → second is unpaired duplicate → skipped
        # So we expect: ACDEF(123), ACDEF(456), GHIKL(nan) = 3 sequences
        assert len(msa.sequences) == 3
