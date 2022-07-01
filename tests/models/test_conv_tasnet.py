import pytest
import torch

from xspeech.models.conv_tasnet import Separator

parameters_separator = [
    (4, 3, 5, 16, 10, 3, 2, 5, "sigmoid", 2, 10),
    (2, 3, 6, 4, 5, 3, 3, 1, "softmax", 3, 16),
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
    mask_nonlinear: bool,
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
