from typing import Tuple

import pytest
import torch

from xspeech.utils import set_seed
from xspeech.models.film import FiLM, FiLM1d

parameters_film = [
    (4, 3, (10,)),
    (2, 4, (5, 6)),
]
parameters_film1d = [
    (4, 3, 10),
]


@pytest.mark.parametrize("batch_size, num_features, shape", parameters_film)
def test_film(batch_size: int, num_features: int, shape: Tuple[int, ...]):
    set_seed()

    model = FiLM()

    input = torch.randn(batch_size, num_features, *shape)
    scale = torch.randn(batch_size, num_features)
    bias = torch.randn(batch_size, num_features)
    output = model(input, scale=scale, bias=bias)

    assert input.size() == output.size()


@pytest.mark.parametrize("batch_size, num_features, num_samples", parameters_film1d)
def test_film1d(batch_size: int, num_features: int, num_samples: int):
    set_seed()

    model = FiLM1d()

    input = torch.randn(batch_size, num_features, num_samples)
    scale = torch.randn(batch_size, num_features)
    bias = torch.randn(batch_size, num_features)
    output = model(input, scale=scale, bias=bias)

    assert input.size() == output.size()
