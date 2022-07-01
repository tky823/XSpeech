from typing import Optional

import pytest
import torch

from xspeech.models.filterbank import Encoder, Decoder

parameters_encoder = [
    (4, 1, 10, 16, 8, "relu", 32),
    (2, 3, 4, 5, 3, None, 16),
]
parameters_decoder = [
    (4, 1, 10, 16, 8, 10),
    (2, 3, 4, 5, 3, 8),
]


@pytest.mark.parametrize(
    "batch_size, in_channels, n_basis, \
        kernel_size, stride, nonlinear, num_samples",
    parameters_encoder,
)
def test_encoder(
    batch_size: int,
    in_channels: int,
    n_basis: int,
    kernel_size: int,
    stride: int,
    nonlinear: Optional[str],
    num_samples: int,
):
    model = Encoder(
        in_channels,
        n_basis=n_basis,
        kernel_size=kernel_size,
        stride=stride,
        nonlinear=nonlinear,
    )

    input = torch.randn(batch_size, in_channels, num_samples)

    with torch.no_grad():
        output = model(input)

    assert output.size()[:2] == (batch_size, n_basis)


@pytest.mark.parametrize(
    "batch_size, out_channels, n_basis, kernel_size, stride, num_samples",
    parameters_decoder,
)
def test_decoder(
    batch_size: int,
    out_channels: int,
    n_basis: int,
    kernel_size: int,
    stride: int,
    num_samples: int,
):
    model = Decoder(
        n_basis,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
    )

    input = torch.randn(batch_size, n_basis, num_samples)

    with torch.no_grad():
        output = model(input)

    assert output.size()[:2] == (batch_size, out_channels)
