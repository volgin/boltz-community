"""Tests for boltz.model.layers.dropout — get_dropout_mask."""

import torch
import pytest

from boltz.model.layers.dropout import get_dropout_mask


class TestGetDropoutMask:
    """Tests for get_dropout_mask."""

    def test_training_false_all_ones(self):
        """training=False → all-ones mask (no dropout)."""
        z = torch.randn(2, 4, 6, 8)
        mask = get_dropout_mask(0.25, z, training=False)
        # dropout * training = 0.25 * 0 = 0 → all pass, scaled by 1/(1-0) = 1
        expected = torch.ones_like(mask)
        assert torch.allclose(mask, expected)

    def test_row_shape(self):
        """Row-wise dropout: mask shape (B, N, 1, 1)."""
        z = torch.randn(2, 4, 6, 8)
        mask = get_dropout_mask(0.25, z, training=True, columnwise=False)
        assert mask.shape == (2, 4, 1, 1)

    def test_column_shape(self):
        """Column-wise dropout: mask shape (B, 1, N, 1)."""
        z = torch.randn(2, 4, 6, 8)
        mask = get_dropout_mask(0.25, z, training=True, columnwise=True)
        assert mask.shape == (2, 1, 6, 1)

    def test_scaling(self):
        """Mask values are either 0 or 1/(1-p)."""
        torch.manual_seed(42)
        p = 0.3
        z = torch.randn(1, 100, 100, 8)
        mask = get_dropout_mask(p, z, training=True, columnwise=False)

        unique = torch.unique(mask)
        # Should only contain 0 and 1/(1-p)
        expected_scale = 1.0 / (1.0 - p)
        for v in unique:
            assert v.item() == pytest.approx(0.0, abs=1e-6) or v.item() == pytest.approx(
                expected_scale, abs=1e-4
            )
