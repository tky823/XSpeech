import pytest
import torch
import torch.nn.functional as F

from xspeech.models.tdcn import (
    TimeDilatedConvBlock1d,
    ConditionedTimeDilatedConvBlock1d,
    ResidualBlock1d,
    ConditionedResidualBlock1d,
    DepthwiseSeparableConv1d,
    ConditionedDepthwiseSeparableConv1d,
)

parameters_time_dilated_conv_block1d = [
    (4, 3, 5, 16, 3, 5, 10),
    (2, 3, 6, 4, 3, 1, 16),
]
parameters_residual_block1d = [
    (4, 3, 5, 16, 3, 1, 1, 10),
    (2, 3, 6, 4, 3, 1, 2, 16),
]
parameters_depthwise_separable_conv1d = [
    (4, 3, 16, 12, 3, 1, 1, 10),
    (2, 3, 4, 5, 3, 1, 2, 16),
]


@pytest.mark.parametrize(
    "batch_size, num_features, hidden_channels, skip_channels, \
        kernel_size, num_layers, num_samples",
    parameters_time_dilated_conv_block1d,
)
def test_time_dilated_conv_block1d(
    batch_size: int,
    num_features: int,
    hidden_channels: int,
    skip_channels: int,
    kernel_size: int,
    num_layers: int,
    num_samples: int,
):
    model = TimeDilatedConvBlock1d(
        num_features,
        hidden_channels=hidden_channels,
        skip_channels=skip_channels,
        kernel_size=kernel_size,
        num_layers=num_layers,
        dilated=True,
        causal=False,
        norm=True,
        dual_head=True,
    )

    input = torch.randn(batch_size, num_features, num_samples)

    with torch.no_grad():
        output, skip = model(input)

    assert output.size() == (batch_size, num_features, num_samples)
    assert skip.size() == (batch_size, skip_channels, num_samples)


@pytest.mark.parametrize(
    "batch_size, num_features, hidden_channels, skip_channels, \
        kernel_size, num_layers, num_samples",
    parameters_time_dilated_conv_block1d,
)
def test_conditioned_time_dilated_conv_block1d(
    batch_size: int,
    num_features: int,
    hidden_channels: int,
    skip_channels: int,
    kernel_size: int,
    num_layers: int,
    num_samples: int,
):
    model = ConditionedTimeDilatedConvBlock1d(
        num_features,
        hidden_channels=hidden_channels,
        skip_channels=skip_channels,
        kernel_size=kernel_size,
        num_layers=num_layers,
        dilated=True,
        causal=False,
        norm=True,
        dual_head=True,
    )

    input = torch.randn(batch_size, num_features, num_samples)
    scale, bias = [], []

    for _ in range(num_layers):
        scale.append(torch.randn(batch_size, hidden_channels))
        bias.append(torch.randn(batch_size, hidden_channels))

    with torch.no_grad():
        output, skip = model(input, scale=scale, bias=bias)

    assert output.size() == (batch_size, num_features, num_samples)
    assert skip.size() == (batch_size, skip_channels, num_samples)


@pytest.mark.parametrize(
    "batch_size, in_channels, hidden_channels, skip_channels, \
        kernel_size, stride, dilation, num_samples",
    parameters_residual_block1d,
)
def test_residual_block1d(
    batch_size: int,
    in_channels: int,
    hidden_channels: int,
    skip_channels: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    num_samples: int,
):
    model = ResidualBlock1d(
        in_channels,
        hidden_channels=hidden_channels,
        skip_channels=skip_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        causal=False,
        norm=True,
        dual_head=True,
    )

    input = torch.randn(batch_size, in_channels, num_samples)

    with torch.no_grad():
        output, skip = model(input)

    assert output.size() == (batch_size, in_channels, num_samples)
    assert skip.size() == (batch_size, skip_channels, num_samples)


@pytest.mark.parametrize(
    "batch_size, in_channels, hidden_channels, skip_channels, \
        kernel_size, stride, dilation, num_samples",
    parameters_residual_block1d,
)
def test_conditioned_residual_block(
    batch_size: int,
    in_channels: int,
    hidden_channels: int,
    skip_channels: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    num_samples: int,
):
    model = ConditionedResidualBlock1d(
        in_channels,
        hidden_channels=hidden_channels,
        skip_channels=skip_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        causal=False,
        norm=True,
        dual_head=True,
    )

    input = torch.randn(batch_size, in_channels, num_samples)
    scale = torch.randn(batch_size, hidden_channels)
    bias = torch.randn(batch_size, hidden_channels)

    with torch.no_grad():
        output, skip = model(input, scale=scale, bias=bias)

    assert output.size() == (batch_size, in_channels, num_samples)
    assert skip.size() == (batch_size, skip_channels, num_samples)


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, skip_channels, \
        kernel_size, stride, dilation, num_samples",
    parameters_depthwise_separable_conv1d,
)
def test_depthwise_separable_conv1d(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    skip_channels: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    num_samples: int,
):
    conv1d = DepthwiseSeparableConv1d(
        in_channels,
        out_channels=out_channels,
        skip_channels=skip_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
    )

    input = torch.randn(batch_size, in_channels, num_samples)

    padding = (
        (num_samples - 1) * stride - num_samples + (kernel_size - 1) * dilation + 1
    )
    padding_left = padding // 2
    padding_right = padding - padding_left

    x = F.pad(input, (padding_left, padding_right))

    with torch.no_grad():
        output, skip = conv1d(x)

    assert output.size() == (batch_size, out_channels, num_samples)
    assert skip.size() == (batch_size, skip_channels, num_samples)


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, skip_channels, \
        kernel_size, stride, dilation, num_samples",
    parameters_depthwise_separable_conv1d,
)
def test_conditioned_depthwise_separable_conv1d(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    skip_channels: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    num_samples: int,
):
    conv1d = ConditionedDepthwiseSeparableConv1d(
        in_channels,
        out_channels=out_channels,
        skip_channels=skip_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
    )

    input = torch.randn(batch_size, in_channels, num_samples)
    scale = torch.randn(batch_size, in_channels)
    bias = torch.randn(batch_size, in_channels)

    padding = (
        (num_samples - 1) * stride - num_samples + (kernel_size - 1) * dilation + 1
    )
    padding_left = padding // 2
    padding_right = padding - padding_left

    x = F.pad(input, (padding_left, padding_right))

    with torch.no_grad():
        output, skip = conv1d(x, scale=scale, bias=bias)

    assert output.size() == (batch_size, out_channels, num_samples)
    assert skip.size() == (batch_size, skip_channels, num_samples)
