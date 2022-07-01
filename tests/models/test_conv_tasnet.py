import pytest
import torch

from xspeech.models.conv_tasnet import Separator, ConditionedExtractor

parameters_separator = [
    (4, 3, 5, 16, 10, 3, 2, 5, "sigmoid", 2, 10),
    (2, 3, 6, 4, 5, 3, 3, 1, "softmax", 3, 16),
]
parameters_conditioned_extractor = [
    (4, 3, 5, 16, 10, 3, 2, 5, 10),
    (2, 3, 6, 4, 5, 3, 3, 1, 16),
]


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
