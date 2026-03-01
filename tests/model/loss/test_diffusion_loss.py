"""Tests for boltz.model.loss.diffusion — weighted_rigid_align and smooth_lddt_loss."""

import torch
import pytest


def _rotation_matrix_z(angle):
    """Return a 3x3 rotation matrix around the Z axis."""
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


class TestWeightedRigidAlign:
    """Tests for weighted_rigid_align."""

    def test_identity(self):
        """pred == true → aligned output equals input."""
        from boltz.model.loss.diffusion import weighted_rigid_align

        B, N = 2, 10
        coords = torch.randn(B, N, 3)
        weights = torch.ones(B, N)
        mask = torch.ones(B, N)

        aligned = weighted_rigid_align(coords, coords, weights, mask)
        assert torch.allclose(aligned, coords, atol=1e-4)

    def test_known_rotation(self):
        """Apply known rotation, alignment should recover original."""
        from boltz.model.loss.diffusion import weighted_rigid_align

        torch.manual_seed(42)
        B, N = 1, 20
        true_coords = torch.randn(B, N, 3)
        R = _rotation_matrix_z(torch.tensor(0.5))
        pred_coords = true_coords @ R.T + torch.tensor([1.0, 2.0, 3.0])
        weights = torch.ones(B, N)
        mask = torch.ones(B, N)

        aligned = weighted_rigid_align(true_coords, pred_coords, weights, mask)
        # After alignment, true_coords should be close to pred_coords
        rmsd = (aligned - pred_coords).pow(2).sum(-1).sqrt().mean()
        assert rmsd < 0.1

    def test_mask_respected(self):
        """Masked atoms should not affect alignment quality."""
        from boltz.model.loss.diffusion import weighted_rigid_align

        torch.manual_seed(7)
        B, N = 1, 20
        true_coords = torch.randn(B, N, 3)
        pred_coords = true_coords.clone()
        weights = torch.ones(B, N)

        # Corrupt last 5 atoms in true_coords
        true_corrupted = true_coords.clone()
        true_corrupted[:, -5:] += 100.0

        # Mask out corrupted atoms
        mask = torch.ones(B, N)
        mask[:, -5:] = 0.0

        aligned = weighted_rigid_align(true_corrupted, pred_coords, weights, mask)
        # Unmasked atoms should still align well
        unmasked_rmsd = (aligned[:, :15] - pred_coords[:, :15]).pow(2).sum(-1).sqrt().mean()
        assert unmasked_rmsd < 0.5

    def test_weights_effect(self):
        """Non-uniform weights should bias the alignment toward high-weight atoms."""
        from boltz.model.loss.diffusion import weighted_rigid_align

        torch.manual_seed(99)
        B, N = 1, 10
        true_coords = torch.randn(B, N, 3)
        R = _rotation_matrix_z(torch.tensor(0.3))
        pred_coords = true_coords @ R.T + torch.tensor([0.5, -0.5, 0.0])
        mask = torch.ones(B, N)

        # High weight on first 3 atoms
        weights_focused = torch.zeros(B, N)
        weights_focused[:, :3] = 10.0

        aligned = weighted_rigid_align(true_coords, pred_coords, weights_focused, mask)
        # RMSD on focused atoms should be small
        focused_rmsd = (aligned[:, :3] - pred_coords[:, :3]).pow(2).sum(-1).sqrt().mean()
        assert focused_rmsd < 0.5


class TestSmoothLddtLoss:
    """Tests for smooth_lddt_loss."""

    def test_perfect_prediction(self):
        """Identical coords → loss ≈ 0 (sigmoid(0.5 - 0) ≈ 0.62, so loss ~ 1 - lddt)."""
        from boltz.model.loss.diffusion import smooth_lddt_loss

        B, N = 2, 10
        coords = torch.randn(B, N, 3)
        is_nuc = torch.zeros(B, N)
        mask = torch.ones(B, N)

        loss = smooth_lddt_loss(coords, coords, is_nuc, mask)
        # Perfect prediction: dist_diff = 0, sigmoids are > 0.5, lddt is high
        assert loss.item() < 0.5

    def test_range(self):
        """Loss should be in [0, 1]."""
        from boltz.model.loss.diffusion import smooth_lddt_loss

        torch.manual_seed(123)
        B, N = 2, 15
        pred = torch.randn(B, N, 3) * 10
        true = torch.randn(B, N, 3) * 10
        is_nuc = torch.zeros(B, N)
        mask = torch.ones(B, N)

        loss = smooth_lddt_loss(pred, true, is_nuc, mask)
        assert 0.0 <= loss.item() <= 1.0

    def test_nucleotide_cutoff(self):
        """Nucleotides use 30A cutoff — more pairs counted than 15A protein cutoff."""
        from boltz.model.loss.diffusion import smooth_lddt_loss

        torch.manual_seed(42)
        B, N = 1, 20
        # Place atoms in a line spaced ~20A apart (between 15 and 30)
        true = torch.zeros(B, N, 3)
        true[0, :, 0] = torch.arange(N).float() * 20.0
        pred = true + torch.randn_like(true) * 0.1
        mask = torch.ones(B, N)

        loss_prot = smooth_lddt_loss(pred, true, torch.zeros(B, N), mask)
        loss_nuc = smooth_lddt_loss(pred, true, torch.ones(B, N), mask)

        # Nucleotides consider more pairs (cutoff 30 vs 15), so loss may differ
        # Both should be finite
        assert torch.isfinite(loss_prot)
        assert torch.isfinite(loss_nuc)

    def test_zero_mask(self):
        """All-zero mask → no NaN/Inf."""
        from boltz.model.loss.diffusion import smooth_lddt_loss

        B, N = 1, 8
        pred = torch.randn(B, N, 3)
        true = torch.randn(B, N, 3)
        is_nuc = torch.zeros(B, N)
        mask = torch.zeros(B, N)

        loss = smooth_lddt_loss(pred, true, is_nuc, mask)
        assert torch.isfinite(loss)
