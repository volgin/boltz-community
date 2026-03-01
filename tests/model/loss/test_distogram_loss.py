"""Tests for boltz.model.loss.distogram — distogram_loss."""

import torch
import pytest


class TestDistogramLoss:
    """Tests for distogram_loss."""

    def test_perfect_prediction(self):
        """Matching logits/targets → loss ≈ 0."""
        from boltz.model.loss.distogram import distogram_loss

        B, N, bins = 2, 6, 39
        # Create one-hot target
        target_idx = torch.randint(0, bins, (B, N, N))
        target = torch.nn.functional.one_hot(target_idx, bins).float()

        # Create logits that strongly match target
        logits = target * 20.0 - 10.0  # large positive for correct bin

        mask = torch.ones(B, N)
        feats = {"token_disto_mask": mask, "disto_target": target}
        output = {"pdistogram": logits}

        global_loss, batch_loss = distogram_loss(output, feats)
        assert global_loss.item() < 0.1

    def test_mask_effect(self):
        """Zero mask → 0 loss."""
        from boltz.model.loss.distogram import distogram_loss

        B, N, bins = 1, 6, 39
        target = torch.randn(B, N, N, bins).softmax(-1)
        logits = torch.randn(B, N, N, bins)
        mask = torch.zeros(B, N)

        feats = {"token_disto_mask": mask, "disto_target": target}
        output = {"pdistogram": logits}

        global_loss, batch_loss = distogram_loss(output, feats)
        assert global_loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_batch_shape(self):
        """batch_loss shape is (B,), global_loss is scalar."""
        from boltz.model.loss.distogram import distogram_loss

        B, N, bins = 3, 8, 39
        target = torch.randn(B, N, N, bins).softmax(-1)
        logits = torch.randn(B, N, N, bins)
        mask = torch.ones(B, N)

        feats = {"token_disto_mask": mask, "disto_target": target}
        output = {"pdistogram": logits}

        global_loss, batch_loss = distogram_loss(output, feats)
        assert batch_loss.shape == (B,)
        assert global_loss.dim() == 0
        assert torch.isfinite(global_loss)
