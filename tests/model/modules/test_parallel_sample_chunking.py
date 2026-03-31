import pytest
import torch


@pytest.mark.parametrize(("multiplicity", "max_parallel_samples"), [(10, 5), (3, 5), (11, 5)])
def test_v1_sampling_respects_max_parallel_samples(
    multiplicity: int,
    max_parallel_samples: int,
    expected_sample_chunk_sizes,
    v1_atom_diffusion_factory,
):
    call_sizes = []
    diffusion = v1_atom_diffusion_factory(call_sizes)

    result = diffusion.sample(
        atom_mask=torch.ones(1, 4),
        multiplicity=multiplicity,
        max_parallel_samples=max_parallel_samples,
        num_sampling_steps=2,
        feats={"token_index": torch.zeros(1, 3, dtype=torch.long)},
        steering_args=None,
    )

    assert result["sample_atom_coords"].shape[0] == multiplicity
    assert call_sizes == expected_sample_chunk_sizes(
        multiplicity, max_parallel_samples, num_steps=2
    )


@pytest.mark.parametrize(("multiplicity", "max_parallel_samples"), [(10, 5), (3, 5), (11, 5)])
def test_v2_sampling_respects_max_parallel_samples(
    multiplicity: int,
    max_parallel_samples: int,
    expected_sample_chunk_sizes,
    v2_atom_diffusion_factory,
):
    call_sizes = []
    diffusion = v2_atom_diffusion_factory(call_sizes)

    result = diffusion.sample(
        atom_mask=torch.ones(1, 4),
        multiplicity=multiplicity,
        max_parallel_samples=max_parallel_samples,
        num_sampling_steps=2,
        feats={},
        steering_args={
            "fk_steering": False,
            "physical_guidance_update": False,
            "contact_guidance_update": False,
        },
    )

    assert result["sample_atom_coords"].shape[0] == multiplicity
    assert call_sizes == expected_sample_chunk_sizes(
        multiplicity, max_parallel_samples, num_steps=2
    )
