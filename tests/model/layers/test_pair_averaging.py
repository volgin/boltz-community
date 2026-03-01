"""Tests for boltz.model.layers.pair_averaging — PairWeightedAveraging."""

import torch
import pytest

from boltz.model.layers.pair_averaging import PairWeightedAveraging


@pytest.fixture
def pair_avg_layer():
    """Create a small PairWeightedAveraging layer."""
    layer = PairWeightedAveraging(c_m=32, c_z=16, c_h=8, num_heads=4)
    layer.eval()
    return layer


class TestPairWeightedAveraging:
    """Tests for PairWeightedAveraging."""

    def test_output_shape(self, pair_avg_layer):
        """Output has same shape as input m."""
        B, S, N, D_m, D_z = 2, 3, 6, 32, 16
        m = torch.randn(B, S, N, D_m)
        z = torch.randn(B, N, N, D_z)
        mask = torch.ones(B, N, N)

        with torch.no_grad():
            out = pair_avg_layer(m, z, mask)
        assert out.shape == (B, S, N, D_m)

    def test_chunk_heads_equivalence(self, pair_avg_layer):
        """chunk_heads=True produces same result as False."""
        B, S, N, D_m, D_z = 1, 2, 6, 32, 16
        m = torch.randn(B, S, N, D_m)
        z = torch.randn(B, N, N, D_z)
        mask = torch.ones(B, N, N)

        with torch.no_grad():
            out_full = pair_avg_layer(m, z, mask, chunk_heads=False)
            out_chunked = pair_avg_layer(m, z, mask, chunk_heads=True)

        assert torch.allclose(out_full, out_chunked, atol=1e-4)

    def test_mask_no_nan(self, pair_avg_layer):
        """Zero mask should not produce NaN."""
        B, S, N, D_m, D_z = 1, 2, 6, 32, 16
        m = torch.randn(B, S, N, D_m)
        z = torch.randn(B, N, N, D_z)
        mask = torch.zeros(B, N, N)

        with torch.no_grad():
            out = pair_avg_layer(m, z, mask)
        assert torch.isfinite(out).all()
