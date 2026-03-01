"""Tests for boltz.model.loss.bfactor — bfactor_loss_fn."""

import torch
import pytest


class TestBfactorLoss:
    """Tests for bfactor_loss_fn."""

    def test_perfect_prediction(self, small_feats_dict):
        """Matching bins → loss ≈ 0."""
        from boltz.model.loss.bfactor import bfactor_loss_fn

        B, N_tok, N_atom = 1, 8, 16
        bins = 20

        # Compute what the target bins would be
        bfactor_atom = small_feats_dict["bfactor"].unsqueeze(-1)
        token_to_rep_atom = small_feats_dict["token_to_rep_atom"].float()
        bfactor_token = torch.bmm(token_to_rep_atom, bfactor_atom)
        boundaries = torch.linspace(0, 100, bins - 1)
        target_bin = (bfactor_token > boundaries).sum(dim=-1).long()

        # Create logits that perfectly match targets
        logits = torch.full((B, N_tok, bins), -10.0)
        for b in range(B):
            for t in range(N_tok):
                logits[b, t, target_bin[b, t]] = 10.0

        output = {"pbfactor": logits}
        loss = bfactor_loss_fn(output, small_feats_dict)
        assert loss.item() < 0.1

    def test_zero_masked(self, small_feats_dict):
        """Atoms with bfactor=0 are masked out."""
        from boltz.model.loss.bfactor import bfactor_loss_fn

        B, N_tok, N_atom = 1, 8, 16
        bins = 20

        # Set all bfactors to 0 (they get masked by token_mask = bfactor > 1e-5)
        feats = {**small_feats_dict, "bfactor": torch.zeros(B, N_atom)}
        logits = torch.randn(B, N_tok, bins)
        output = {"pbfactor": logits}

        loss = bfactor_loss_fn(output, feats)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_finiteness(self, small_feats_dict):
        """Random inputs → finite loss."""
        from boltz.model.loss.bfactor import bfactor_loss_fn

        B, N_tok = 1, 8
        bins = 20
        logits = torch.randn(B, N_tok, bins)
        output = {"pbfactor": logits}

        loss = bfactor_loss_fn(output, small_feats_dict)
        assert torch.isfinite(loss)
