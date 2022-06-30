import pytest
import torch
import torch.nn.functional as F

from xspeech.models.tdcn import (
    DepthwiseSeparableConv1d,
    ConditionedDepthwiseSeparableConv1d,
)

parameters_depthwise_separable_conv1d = [
    (4, 3, 16, 12, 3, 1, 1, 10),
    (2, 3, 4, 5, 3, 1, 2, 16),
]


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