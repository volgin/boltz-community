"""Tests for boltz.model.layers.transition — Transition."""

import torch
import pytest

from boltz.model.layers.transition import Transition


@pytest.fixture
def transition_layer():
    """Create a small Transition layer."""
    layer = Transition(dim=32, hidden=64)
    layer.eval()
    return layer


class TestTransition:
    """Tests for Transition."""

    def test_output_shape(self, transition_layer):
        """Output shape matches input shape."""
        B, N, D = 2, 8, 32
        x = torch.randn(B, N, D)

        with torch.no_grad():
            out = transition_layer(x)
        assert out.shape == (B, N, D)

    def test_chunk_equivalence(self, transition_layer):
        """Chunked computation produces same result as non-chunked."""
        B, N, D = 1, 8, 32
        x = torch.randn(B, N, D)

        with torch.no_grad():
            out_full = transition_layer(x, chunk_size=None)
            out_chunked = transition_layer(x, chunk_size=16)

        assert torch.allclose(out_full, out_chunked, atol=1e-5)

    def test_final_init_zero_output(self):
        """Fresh layer with final_init on fc3 → output is zero for zero-norm input."""
        layer = Transition(dim=16, hidden=32)
        layer.eval()

        # fc3 is initialized with final_init (zeros)
        # So fc3(anything) = 0
        B, N, D = 1, 4, 16
        x = torch.randn(B, N, D)

        with torch.no_grad():
            out = layer(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)
