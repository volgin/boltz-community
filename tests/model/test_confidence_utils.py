import math

import torch

from boltz.model.layers.confidence_utils import compute_pae_summaries


def test_compute_pae_summaries_are_contact_weighted_and_nan_when_undefined():
    pae = torch.tensor(
        [
            [
                [0.0, 2.0, 4.0, 20.0],
                [6.0, 0.0, 8.0, 22.0],
                [10.0, 12.0, 0.0, 24.0],
                [26.0, 28.0, 30.0, 0.0],
            ]
        ]
    )
    asym_id = torch.tensor([[0, 0, 1, 1]])
    token_pad_mask = torch.tensor([[True, True, True, False]])

    pred_distogram_logits = torch.full((1, 4, 4, 64), -20.0)
    pred_distogram_logits[..., 20] = 20.0
    for i, j in [(0, 1), (1, 0), (0, 2), (2, 0)]:
        pred_distogram_logits[0, i, j, 20] = -20.0
        pred_distogram_logits[0, i, j, 0] = 20.0

    feats = {
        "asym_id": asym_id,
        "token_pad_mask": token_pad_mask,
    }

    complex_pae, complex_ipae, chains_pae, pair_chains_pae = compute_pae_summaries(
        pae, pred_distogram_logits, feats, multiplicity=1
    )

    assert torch.allclose(complex_pae, torch.tensor([5.5]))
    assert torch.allclose(complex_ipae, torch.tensor([7.0]))
    assert torch.allclose(chains_pae[0], torch.tensor([4.0]))
    assert math.isnan(chains_pae[1].item())
    assert torch.allclose(pair_chains_pae[0][0], torch.tensor([4.0]))
    assert torch.allclose(pair_chains_pae[0][1], torch.tensor([4.0]))
    assert torch.allclose(pair_chains_pae[1][0], torch.tensor([10.0]))
    assert math.isnan(pair_chains_pae[1][1].item())
