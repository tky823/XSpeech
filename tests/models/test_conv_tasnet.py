from typing import Optional
import pytest
import torch

from xspeech.models.conv_tasnet import ConvTasNet, Separator, ConditionedExtractor

parameters_conv_tasnet = [
    (4, 16, 16, 8, "relu", 5, 6, 7, 3, 2, 4, "sigmoid", 2, 64),
    (2, 10, 10, 5, None, 5, 6, 7, 3, 1, 3, "softmax", 3, 64),
]
parameters_separator = [
    (4, 3, 5, 16, 10, 3, 2, 5, "sigmoid", 2, 10),
    (2, 3, 6, 4, 5, 3, 3, 1, "softmax", 3, 16),
]
parameters_conditioned_extractor = [
    (4, 3, 5, 16, 10, 3, 2, 5, 10),
    (2, 3, 6, 4, 5, 3, 3, 1, 16),
]


@pytest.mark.parametrize(
    "batch_size, n_basis, kernel_size, stride, enc_nonlinear,\
        sep_hidden_channels, sep_bottleneck_channels, sep_skip_channels, \
        sep_kernel_size, sep_num_blocks, sep_num_layers, mask_nonlinear, \
        n_sources, num_samples",
    parameters_conv_tasnet,
)
def test_conv_tasnet(
    batch_size: int,
    n_basis: int,
    kernel_size: int,
    stride: int,
    enc_nonlinear: Optional[str],
    sep_hidden_channels: int,
    sep_bottleneck_channels: int,
    sep_skip_channels: int,
    sep_kernel_size: int,
    sep_num_blocks: int,
    sep_num_layers: int,
    mask_nonlinear: str,
    n_sources: int,
    num_samples: int,
):
    model = ConvTasNet(
        n_basis,
        kernel_size=kernel_size,
        stride=stride,
        enc_nonlinear=enc_nonlinear,
        sep_hidden_channels=sep_hidden_channels,
        sep_bottleneck_channels=sep_bottleneck_channels,
        sep_skip_channels=sep_skip_channels,
        sep_kernel_size=sep_kernel_size,
        sep_num_blocks=sep_num_blocks,
        sep_num_layers=sep_num_layers,
        dilated=True,
        causal=False,
        norm=True,
        mask_nonlinear=mask_nonlinear,
        n_sources=n_sources,
    )

    input = torch.randn(batch_size, 1, num_samples)

    with torch.no_grad():
        output = model(input)

    assert output.size() == (batch_size, n_sources, num_samples)


@pytest.mark.parametrize(
    "batch_size, num_features, bottleneck_channels, hidden_channels, skip_channels, \
        kernel_size, num_blocks, num_layers, mask_nonlinear, n_sources, num_samples",
    parameters_separator,
)
def test_separator(
    batch_size: int,
    num_features: int,
    bottleneck_channels: int,
    hidden_channels: int,
    skip_channels: int,
    kernel_size: int,
    num_blocks: int,
    num_layers: int,
    mask_nonlinear: str,
    n_sources: int,
    num_samples: int,
):
    model = Separator(
        num_features,
        bottleneck_channels=bottleneck_channels,
        hidden_channels=hidden_channels,
        skip_channels=skip_channels,
        kernel_size=kernel_size,
        num_blocks=num_blocks,
        num_layers=num_layers,
        dilated=True,
        causal=False,
        norm=True,
        mask_nonlinear=mask_nonlinear,
        n_sources=n_sources,
    )

    input = torch.randn(batch_size, num_features, num_samples)

    with torch.no_grad():
        output = model(input)

    assert output.size() == (batch_size, n_sources, num_features, num_samples)


@pytest.mark.parametrize(
    "batch_size, num_features, bottleneck_channels, hidden_channels, skip_channels, \
        kernel_size, num_blocks, num_layers, num_samples",
    parameters_conditioned_extractor,
)
def test_conditioned_extractor(
    batch_size: int,
    num_features: int,
    bottleneck_channels: int,
    hidden_channels: int,
    skip_channels: int,
    kernel_size: int,
    num_blocks: int,
    num_layers: int,
    num_samples: int,
):
    model = ConditionedExtractor(
        num_features,
        bottleneck_channels=bottleneck_channels,
        hidden_channels=hidden_channels,
        skip_channels=skip_channels,
        kernel_size=kernel_size,
        num_blocks=num_blocks,
        num_layers=num_layers,
        dilated=True,
        causal=False,
        norm=True,
    )

    input = torch.randn(batch_size, num_features, num_samples)
    scale, bias = [], []

    for _ in range(num_blocks):
        _scale, _bias = [], []

        for _ in range(num_layers):
            _scale.append(torch.randn(batch_size, hidden_channels))
            _bias.append(torch.randn(batch_size, hidden_channels))

        scale.append(_scale)
        bias.append(_bias)

    with torch.no_grad():
        output = model(input, scale=scale, bias=bias)

    assert input.size() == output.size()
