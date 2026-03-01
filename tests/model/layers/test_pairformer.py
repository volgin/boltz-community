"""Tests for boltz.model.layers.pairformer — PairformerLayer, PairformerNoSeqLayer."""

import torch
import pytest


class TestPairformerLayer:
    """Tests for PairformerLayer."""

    def test_output_shapes(self):
        """Layer output shapes match input shapes."""
        from boltz.model.layers.pairformer import PairformerLayer

        token_s, token_z = 64, 32
        layer = PairformerLayer(
            token_s=token_s, token_z=token_z,
            num_heads=4, dropout=0.0,
            pairwise_head_width=8, pairwise_num_heads=2,
        )
        layer.eval()

        B, N = 1, 8
        s = torch.randn(B, N, token_s)
        z = torch.randn(B, N, N, token_z)
        mask = torch.ones(B, N)
        pair_mask = torch.ones(B, N, N)

        with torch.no_grad():
            s_out, z_out = layer(s, z, mask, pair_mask)

        assert s_out.shape == (B, N, token_s)
        assert z_out.shape == (B, N, N, token_z)

    def test_module_stacking(self):
        """PairformerModule stacks multiple layers."""
        from boltz.model.layers.pairformer import PairformerModule

        token_s, token_z = 64, 32
        module = PairformerModule(
            token_s=token_s, token_z=token_z, num_blocks=2,
            num_heads=4, dropout=0.0,
            pairwise_head_width=8, pairwise_num_heads=2,
        )
        module.eval()

        B, N = 1, 8
        s = torch.randn(B, N, token_s)
        z = torch.randn(B, N, N, token_z)
        mask = torch.ones(B, N)
        pair_mask = torch.ones(B, N, N)

        with torch.no_grad():
            s_out, z_out = module(s, z, mask, pair_mask)

        assert s_out.shape == (B, N, token_s)
        assert z_out.shape == (B, N, N, token_z)

    def test_no_seq_variant(self):
        """PairformerNoSeqLayer produces valid z output."""
        from boltz.model.layers.pairformer import PairformerNoSeqLayer

        token_z = 32
        layer = PairformerNoSeqLayer(
            token_z=token_z, dropout=0.0,
            pairwise_head_width=8, pairwise_num_heads=2,
        )
        layer.eval()

        B, N = 1, 8
        z = torch.randn(B, N, N, token_z)
        pair_mask = torch.ones(B, N, N)

        with torch.no_grad():
            z_out = layer(z, pair_mask)

        assert z_out.shape == (B, N, N, token_z)
        assert torch.isfinite(z_out).all()

    def test_eval_determinism(self):
        """Eval mode produces deterministic results (dropout=0)."""
        from boltz.model.layers.pairformer import PairformerLayer

        token_s, token_z = 64, 32
        layer = PairformerLayer(
            token_s=token_s, token_z=token_z,
            num_heads=4, dropout=0.0,
            pairwise_head_width=8, pairwise_num_heads=2,
        )
        layer.eval()

        B, N = 1, 6
        s = torch.randn(B, N, token_s)
        z = torch.randn(B, N, N, token_z)
        mask = torch.ones(B, N)
        pair_mask = torch.ones(B, N, N)

        with torch.no_grad():
            s1, z1 = layer(s, z, mask, pair_mask)
            s2, z2 = layer(s, z, mask, pair_mask)

        assert torch.allclose(s1, s2, atol=1e-6)
        assert torch.allclose(z1, z2, atol=1e-6)
