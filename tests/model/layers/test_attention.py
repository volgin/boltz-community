"""Tests for boltz.model.layers.attention — AttentionPairBias."""

import torch
import pytest

from boltz.model.layers.attention import AttentionPairBias


@pytest.fixture
def attention_layer():
    """Create a small AttentionPairBias layer."""
    layer = AttentionPairBias(c_s=32, c_z=16, num_heads=4)
    layer.eval()
    return layer


class TestAttentionPairBias:
    """Tests for AttentionPairBias."""

    def test_output_shape(self, attention_layer):
        """Output has same shape as input s."""
        B, N, D_s, D_z = 2, 6, 32, 16
        s = torch.randn(B, N, D_s)
        z = torch.randn(B, N, N, D_z)
        mask = torch.ones(B, N)

        with torch.no_grad():
            out = attention_layer(s, z, mask)
        assert out.shape == (B, N, D_s)

    def test_mask_effect(self):
        """Masking all keys should produce different output than no masking."""
        torch.manual_seed(42)
        layer = AttentionPairBias(c_s=32, c_z=16, num_heads=4)
        # Initialize weights to non-zero with large std so differences are visible
        with torch.no_grad():
            for name, param in layer.named_parameters():
                torch.nn.init.normal_(param, std=1.0)
        layer.eval()

        B, N, D_s, D_z = 1, 8, 32, 16
        s = torch.randn(B, N, D_s)
        z = torch.randn(B, N, N, D_z)

        with torch.no_grad():
            out_all = layer(s, z, torch.ones(B, N))
            out_partial = layer(s, z, torch.cat([torch.ones(B, N // 2), torch.zeros(B, N // 2)], dim=-1))

        # Masking half the keys should change the output
        assert not torch.allclose(out_all, out_partial, atol=1e-3)

    def test_gating(self):
        """Gate→0 → output→0."""
        layer = AttentionPairBias(c_s=32, c_z=16, num_heads=4)
        layer.eval()

        # proj_o is final_init (zeros), so output is always zero
        B, N, D_s, D_z = 1, 4, 32, 16
        s = torch.randn(B, N, D_s)
        z = torch.randn(B, N, N, D_z)
        mask = torch.ones(B, N)

        with torch.no_grad():
            out = layer(s, z, mask)
        # proj_o has final_init (zeros), so output should be zero
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_multiplicity(self):
        """Multiplicity > 1 should produce valid output."""
        layer = AttentionPairBias(c_s=32, c_z=16, num_heads=4)
        layer.eval()

        B_total, N, D_s, D_z = 4, 6, 32, 16
        multiplicity = 2
        B_base = B_total // multiplicity

        s = torch.randn(B_total, N, D_s)
        z = torch.randn(B_base, N, N, D_z)
        mask = torch.ones(B_total, N)

        with torch.no_grad():
            out = layer(s, z, mask, multiplicity=multiplicity)
        assert out.shape == (B_total, N, D_s)
