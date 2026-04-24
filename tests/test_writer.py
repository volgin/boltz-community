import torch

from boltz.data.write.writer import _select_prediction_value


def test_select_prediction_value_serializes_nan_as_null():
    summary = {
        0: torch.tensor([float("nan")]),
        1: {
            0: torch.tensor([1.5]),
            1: torch.tensor([2.5]),
        },
    }

    assert _select_prediction_value(summary, 0) == {
        0: None,
        1: {
            0: 1.5,
            1: 2.5,
        },
    }
