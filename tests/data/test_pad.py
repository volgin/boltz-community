"""Tests for boltz.data.pad — pad_dim and pad_to_max."""

import torch
import pytest

from boltz.data.pad import pad_dim, pad_to_max


class TestPadDim:
    """Tests for pad_dim."""

    def test_basic(self):
        """Pad along dim 1 extends the size."""
        x = torch.randn(2, 5, 3)
        padded = pad_dim(x, dim=1, pad_len=3)
        assert padded.shape == (2, 8, 3)
        # Original data preserved
        assert torch.allclose(padded[:, :5, :], x)
        # Padded region is zero
        assert torch.allclose(padded[:, 5:, :], torch.zeros(2, 3, 3))

    def test_zero_pad(self):
        """pad_len=0 returns input unchanged."""
        x = torch.randn(3, 4)
        padded = pad_dim(x, dim=0, pad_len=0)
        assert torch.allclose(padded, x)

    def test_custom_value(self):
        """Custom pad value is respected."""
        x = torch.zeros(2, 3)
        padded = pad_dim(x, dim=1, pad_len=2, value=-1.0)
        assert padded.shape == (2, 5)
        assert torch.allclose(padded[:, 3:], torch.full((2, 2), -1.0))

    def test_first_dim(self):
        """Pad along dim 0."""
        x = torch.randn(3, 4, 5)
        padded = pad_dim(x, dim=0, pad_len=2)
        assert padded.shape == (5, 4, 5)

    def test_last_dim(self):
        """Pad along last dimension."""
        x = torch.randn(2, 3, 4)
        padded = pad_dim(x, dim=2, pad_len=1)
        assert padded.shape == (2, 3, 5)


class TestPadToMax:
    """Tests for pad_to_max."""

    def test_same_shapes(self):
        """All same shapes → stacked, no padding needed."""
        tensors = [torch.randn(4, 3), torch.randn(4, 3)]
        data, padding = pad_to_max(tensors)
        assert data.shape == (2, 4, 3)
        # padding is 0 (scalar) when shapes match
        assert padding == 0

    def test_different_shapes(self):
        """Different shapes → padded to max, mask returned."""
        t1 = torch.ones(3, 2)
        t2 = torch.ones(5, 4)
        data, padding = pad_to_max([t1, t2])
        assert data.shape == (2, 5, 4)
        # Padding mask shows which elements are original
        assert padding.shape == (2, 5, 4)
        # t1 occupies (3, 2), t2 occupies (5, 4)
        assert padding[0, 0, 0].item() == 1.0
        assert padding[0, 4, 0].item() == 0.0  # padded row for t1
