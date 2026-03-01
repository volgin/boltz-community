"""Tests for boltz.data.write.utils — generate_tags."""

import pytest

from boltz.data.write.utils import generate_tags


class TestGenerateTags:
    """Tests for generate_tags."""

    def test_first_26(self):
        """First 26 tags are A-Z."""
        gen = generate_tags()
        first_26 = [next(gen) for _ in range(26)]
        assert first_26[0] == "A"
        assert first_26[25] == "Z"

    def test_two_letter_combos(self):
        """After Z, tags become two-letter combinations."""
        gen = generate_tags()
        # Skip first 26 single-letter tags
        tags = [next(gen) for _ in range(28)]
        # 27th tag should be AA
        assert tags[26] == "AA"
        # 28th should be BA
        assert tags[27] == "BA"

    def test_no_duplicates(self):
        """First 100 tags have no duplicates."""
        gen = generate_tags()
        tags = [next(gen) for _ in range(100)]
        assert len(tags) == len(set(tags))
