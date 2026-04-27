import pytest
import torch

from boltz.model.modules.diffusionv2 import _get_sample_id_chunks


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


def test_v1_sampling_defaults_max_parallel_samples_to_multiplicity(
    v1_atom_diffusion_factory,
):
    multiplicity = 4
    call_sizes = []
    diffusion = v1_atom_diffusion_factory(call_sizes)

    result = diffusion.sample(
        atom_mask=torch.ones(1, 4),
        multiplicity=multiplicity,
        max_parallel_samples=None,
        num_sampling_steps=2,
        feats={"token_index": torch.zeros(1, 3, dtype=torch.long)},
        steering_args=None,
    )

    assert result["sample_atom_coords"].shape[0] == multiplicity
    assert call_sizes == [multiplicity, multiplicity]


def test_v2_sample_id_chunks_are_record_major():
    chunks = _get_sample_id_chunks(
        batch_size=2,
        multiplicity=3,
        max_parallel_samples=2,
        device=torch.device("cpu"),
    )

    assert [chunk.tolist() for chunk in chunks] == [[0, 1, 3, 4], [2, 5]]


def test_v2_sampling_defaults_max_parallel_samples_to_multiplicity(
    v2_atom_diffusion_factory,
):
    multiplicity = 4
    call_sizes = []
    diffusion = v2_atom_diffusion_factory(call_sizes)

    result = diffusion.sample(
        atom_mask=torch.ones(1, 4),
        multiplicity=multiplicity,
        max_parallel_samples=None,
        num_sampling_steps=2,
        feats={},
        steering_args={
            "fk_steering": False,
            "physical_guidance_update": False,
            "contact_guidance_update": False,
        },
    )

    assert result["sample_atom_coords"].shape[0] == multiplicity
    assert call_sizes == [multiplicity, multiplicity]
