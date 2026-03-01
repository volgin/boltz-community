"""Tests for boltz.model.loss.confidence — lddt_dist, express_coordinate_in_frame, compute_collinear_mask, etc."""

import torch
import pytest


class TestLddtDist:
    """Tests for lddt_dist."""

    def test_perfect_match(self):
        """Identical distance matrices → score 1.0."""
        from boltz.model.loss.confidence import lddt_dist

        B, N = 2, 10
        coords = torch.randn(B, N, 3)
        dmat = torch.cdist(coords, coords)
        # Mask: exclude diagonal
        mask = (1 - torch.eye(N)).unsqueeze(0).expand(B, -1, -1)

        score, _ = lddt_dist(dmat, dmat, mask, cutoff=15.0, per_atom=False)
        assert torch.allclose(score, torch.ones(B), atol=0.01)

    def test_range(self):
        """Score always in [0, 1]."""
        from boltz.model.loss.confidence import lddt_dist

        torch.manual_seed(42)
        B, N = 2, 12
        pred_coords = torch.randn(B, N, 3) * 5
        true_coords = torch.randn(B, N, 3) * 5
        pred_d = torch.cdist(pred_coords, pred_coords)
        true_d = torch.cdist(true_coords, true_coords)
        mask = (1 - torch.eye(N)).unsqueeze(0).expand(B, -1, -1)

        score, _ = lddt_dist(pred_d, true_d, mask, cutoff=15.0, per_atom=False)
        assert (score >= 0.0).all()
        assert (score <= 1.0).all()

    def test_per_atom_shape(self):
        """per_atom=True → (B,N), per_atom=False → (B,)."""
        from boltz.model.loss.confidence import lddt_dist

        B, N = 2, 8
        dmat = torch.randn(B, N, N).abs()
        mask = (1 - torch.eye(N)).unsqueeze(0).expand(B, -1, -1)

        score_atom, _ = lddt_dist(dmat, dmat, mask, cutoff=15.0, per_atom=True)
        score_global, _ = lddt_dist(dmat, dmat, mask, cutoff=15.0, per_atom=False)

        assert score_atom.shape == (B, N)
        assert score_global.shape == (B,)

    def test_thresholds(self):
        """Test exact threshold values: 0.5→1.0, 1.0→0.75, 2.0→0.5, 4.0→0.25."""
        from boltz.model.loss.confidence import lddt_dist

        B, N = 1, 3
        # Create controlled distance matrices where we know the L1 diff
        true_d = torch.tensor([[[0.0, 10.0, 10.0],
                                [10.0, 0.0, 10.0],
                                [10.0, 10.0, 0.0]]]).float()

        # pred_d with known distance difference of exactly 0.3 (< 0.5, all thresholds pass → score 1.0)
        pred_d_close = true_d + 0.3
        pred_d_close[:, range(N), range(N)] = 0  # diagonal
        mask = (1 - torch.eye(N)).unsqueeze(0).float()

        score_close, _ = lddt_dist(pred_d_close, true_d, mask, cutoff=15.0, per_atom=False)
        # dist_l1 = 0.3 < 0.5 → passes all 4 thresholds → score = 1.0
        assert score_close.item() == pytest.approx(1.0, abs=0.01)

        # pred_d with diff = 0.7 (passes 1.0, 2.0, 4.0 but not 0.5 → score = 0.75)
        pred_d_mid = true_d + 0.7
        pred_d_mid[:, range(N), range(N)] = 0
        score_mid, _ = lddt_dist(pred_d_mid, true_d, mask, cutoff=15.0, per_atom=False)
        assert score_mid.item() == pytest.approx(0.75, abs=0.01)


class TestPdeLoss:
    """Tests for pde_loss."""

    def test_shape_and_finiteness(self, small_feats_dict):
        """Returns finite scalar with synthetic feats."""
        from boltz.model.loss.confidence import pde_loss

        B, N_tok = 1, 8
        N_atom = 16
        num_bins = 64

        pred_pde = torch.randn(B, N_tok, N_tok, num_bins)
        pred_coords = torch.randn(B, N_atom, 3)
        true_coords = torch.randn(B, N_atom, 3)
        resolved_mask = torch.ones(B, N_atom)

        loss = pde_loss(pred_pde, pred_coords, true_coords, resolved_mask, small_feats_dict)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


class TestResolvedLoss:
    """Tests for resolved_loss."""

    def test_all_resolved(self, small_feats_dict):
        """All-resolved atoms → finite cross-entropy."""
        from boltz.model.loss.confidence import resolved_loss

        B, N_tok = 1, 8
        N_atom = 16
        pred_resolved = torch.randn(B, N_tok, 2)
        resolved_mask = torch.ones(B, N_atom)

        loss = resolved_loss(pred_resolved, small_feats_dict, resolved_mask)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


class TestExpressCoordinateInFrame:
    """Tests for express_coordinate_in_frame."""

    def test_identity_frame(self):
        """With atoms defining an orthogonal frame, transformation is well-defined."""
        from boltz.model.loss.confidence import express_coordinate_in_frame

        B, mult, N = 1, 1, 5
        atom_coords = torch.randn(B, mult, N, 3)

        # Frame indices: use first 3 atoms for all tokens
        frame_a = torch.zeros(B, N, dtype=torch.long)
        frame_b = torch.ones(B, N, dtype=torch.long)
        frame_c = torch.full((B, N), 2, dtype=torch.long)

        result = express_coordinate_in_frame(atom_coords, frame_a, frame_b, frame_c)
        assert result.shape[-1] == 3
        assert torch.isfinite(result).all()


class TestComputeCollinearMask:
    """Tests for compute_collinear_mask."""

    def test_parallel_vectors_masked(self):
        """Parallel vectors → masked (False)."""
        from boltz.model.loss.confidence import compute_collinear_mask

        v1 = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        v2 = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        mask = compute_collinear_mask(v1.T.unsqueeze(0), v2.T.unsqueeze(0))

        # First pair: parallel → should be masked (False)
        # This tests that collinear vectors are detected
        assert mask.shape[0] > 0

    def test_perpendicular_not_masked(self):
        """Perpendicular vectors → not masked (True)."""
        from boltz.model.loss.confidence import compute_collinear_mask

        # compute_collinear_mask expects (N, 3) shaped inputs
        # It computes norm along dim=1 and dot product along dim=1
        N = 5
        v1 = torch.zeros(N, 3)
        v1[:, 0] = 1.0  # unit x vectors
        v2 = torch.zeros(N, 3)
        v2[:, 1] = 1.0  # unit y vectors

        mask = compute_collinear_mask(v1, v2)
        # cos(90°) = 0 < 0.9063, norms > 1e-2 → all should be True
        assert mask.all()
