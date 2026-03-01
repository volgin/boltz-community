"""Tests for boltz.model.layers.initialize — weight initialization functions."""

import torch
import pytest

import boltz.model.layers.initialize as init


class TestInitialize:
    """Tests for initialization functions."""

    def test_final_init_zeros(self):
        """final_init fills weights with zeros."""
        w = torch.randn(32, 32)
        init.final_init_(w)
        assert torch.allclose(w, torch.zeros_like(w))

    def test_gating_init_zeros(self):
        """gating_init fills weights with zeros."""
        w = torch.randn(16, 16)
        init.gating_init_(w)
        assert torch.allclose(w, torch.zeros_like(w))

    def test_lecun_normal_stats(self):
        """lecun_normal_init produces approximately zero-mean values with bounded variance."""
        torch.manual_seed(42)
        fan_in = 256
        w = torch.empty(128, fan_in)
        init.lecun_normal_init_(w)

        # Mean should be close to 0
        assert abs(w.mean().item()) < 0.05
        # Variance should be approximately 1/fan_in
        expected_var = 1.0 / fan_in
        assert abs(w.var().item() - expected_var) < 0.01

    def test_bias_init_values(self):
        """bias_init_zero_ → 0, bias_init_one_ → 1, ipa_point_weights → 0.541..."""
        b_zero = torch.randn(32)
        init.bias_init_zero_(b_zero)
        assert torch.allclose(b_zero, torch.zeros_like(b_zero))

        b_one = torch.randn(32)
        init.bias_init_one_(b_one)
        assert torch.allclose(b_one, torch.ones_like(b_one))

        b_ipa = torch.randn(8)
        init.ipa_point_weights_init_(b_ipa)
        assert torch.allclose(b_ipa, torch.full_like(b_ipa, 0.541324854612918), atol=1e-6)
