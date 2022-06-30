from typing import Tuple

import pytest
import torch

from xspeech.modules.norm import GlobalLayerNorm

parameters_global_norm = [
    (4, 3, (10,)),
    (2, 8, (5, 12)),
]


@pytest.mark.parametrize("batch_size, num_features, shape", parameters_global_norm)
def test_global_norm(batch_size: int, num_features: int, shape: Tuple[int, ...]):
    normNd = GlobalLayerNorm(num_features)

    input = torch.randn(batch_size, num_features, *shape)
    output = normNd(input)

    print(normNd)

    assert input.size() == output.size()
