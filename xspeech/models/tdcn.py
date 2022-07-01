from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .film import FiLM1d
from ..modules.utils.tasnet import choose_layer_norm

EPS = 1e-12


class ResidualBlock1d(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 256,
        skip_channels: int = 256,
        kernel_size: int = 3,
        stride: int = 2,
        dilation: int = 1,
        causal: bool = True,
        nonlinear: Optional[str] = None,
        norm: bool = True,
        dual_head: bool = True,
        eps: float = EPS,
    ):
        super().__init__()

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.causal = causal
        self.norm = norm
        self.dual_head = dual_head

        self.bottleneck_conv1d = nn.Conv1d(
            num_features, hidden_channels, kernel_size=1, stride=1
        )

        if nonlinear is not None:
            if nonlinear == "prelu":
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))

            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
            norm_name = "cLN" if causal else "gLN"
            self.norm1d = choose_layer_norm(
                norm_name, hidden_channels, causal=causal, eps=eps
            )

        self.separable_conv1d = DepthwiseSeparableConv1d(
            hidden_channels,
            num_features,
            skip_channels=skip_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            causal=causal,
            nonlinear=nonlinear,
            norm=norm,
            dual_head=dual_head,
            eps=eps,
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, num_features, num_samples).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, num_features, num_samples).
            torch.Tensor:
                Output tensor for skip connection.
                The shape is (batch_size, skip_channels, num_samples).
        """
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        nonlinear, norm = self.nonlinear, self.norm
        causal = self.causal

        _, _, T = input.size()

        residual = input
        x = self.bottleneck_conv1d(input)

        if nonlinear:
            x = self.nonlinear1d(x)

        if norm:
            x = self.norm1d(x)

        padding = (T - 1) * stride - T + (kernel_size - 1) * dilation + 1

        if causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        x = F.pad(x, (padding_left, padding_right))
        output, skip = self.separable_conv1d(x)

        if output is not None:
            output = output + residual

        return output, skip


class ConditionedResidualBlock1d(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 256,
        skip_channels: int = 256,
        kernel_size: int = 3,
        stride: int = 2,
        dilation: int = 1,
        causal: bool = True,
        nonlinear: Optional[str] = None,
        norm: bool = True,
        dual_head: bool = True,
        eps: float = EPS,
    ):
        super().__init__()

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.causal = causal
        self.norm = norm
        self.dual_head = dual_head

        self.bottleneck_conv1d = nn.Conv1d(
            num_features, hidden_channels, kernel_size=1, stride=1
        )

        if nonlinear is not None:
            if nonlinear == "prelu":
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))

            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
            norm_name = "cLN" if causal else "gLN"
            self.norm1d = choose_layer_norm(
                norm_name, hidden_channels, causal=causal, eps=eps
            )

        self.separable_conv1d = ConditionedDepthwiseSeparableConv1d(
            hidden_channels,
            num_features,
            skip_channels=skip_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            causal=causal,
            nonlinear=nonlinear,
            norm=norm,
            dual_head=dual_head,
            eps=eps,
        )

    def forward(
        self, input: torch.Tensor, scale: torch.Tensor = None, bias: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, num_features, num_samples).
            scale (torch.Tensor):
                Input tensor with shape of (batch_size, num_features).
            bias (torch.Tensor):
                Input tensor with shape of (batch_size, num_features).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, num_features, num_samples).
            torch.Tensor:
                Output tensor for skip connection.
                The shape is (batch_size, skip_channels, num_samples).
        """
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        nonlinear, norm = self.nonlinear, self.norm
        causal = self.causal

        _, _, T = input.size()

        residual = input
        x = self.bottleneck_conv1d(input)

        if nonlinear:
            x = self.nonlinear1d(x)

        if norm:
            x = self.norm1d(x)

        padding = (T - 1) * stride - T + (kernel_size - 1) * dilation + 1

        if causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        x = F.pad(x, (padding_left, padding_right))
        output, skip = self.separable_conv1d(x, scale=scale, bias=bias)

        if output is not None:
            output = output + residual

        return output, skip


class DepthwiseSeparableConv1d(nn.Module):
    r"""Depthwise separable 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        skip_channels: int = 256,
        kernel_size: int = 3,
        stride: int = 2,
        dilation: int = 1,
        causal: bool = False,
        nonlinear: Optional[str] = None,
        norm: bool = True,
        dual_head: bool = True,
        eps: float = EPS,
    ) -> None:
        super().__init__()

        self.dual_head = dual_head
        self.norm = norm
        self.eps = eps

        self.depthwise_conv1d = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
        )

        if nonlinear is not None:
            if nonlinear == "prelu":
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))

            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
            norm_name = "cLN" if causal else "gLN"
            self.norm1d = choose_layer_norm(
                norm_name, in_channels, causal=causal, eps=eps
            )

        if dual_head:
            self.output_pointwise_conv1d = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1
            )

        self.skip_pointwise_conv1d = nn.Conv1d(
            in_channels, skip_channels, kernel_size=1, stride=1
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, in_channels, num_samples).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, in_channels, num_samples).
            torch.Tensor:
                Output tensor for skip connection.
                The shape is (batch_size, in_channels, num_samples).
        """
        nonlinear, norm = self.nonlinear, self.norm
        dual_head = self.dual_head

        x = self.depthwise_conv1d(input)

        if nonlinear:
            x = self.nonlinear1d(x)

        if norm:
            x = self.norm1d(x)

        if dual_head:
            output = self.output_pointwise_conv1d(x)
        else:
            output = None

        skip = self.skip_pointwise_conv1d(x)

        return output, skip


class ConditionedDepthwiseSeparableConv1d(nn.Module):
    r"""Depthwise separable 1D convolution using FiLM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        skip_channels: int = 256,
        kernel_size: int = 3,
        stride: int = 2,
        dilation: int = 1,
        causal: bool = False,
        nonlinear: Optional[str] = None,
        norm: bool = True,
        dual_head: bool = True,
        eps: float = EPS,
    ) -> None:
        super().__init__()

        self.dual_head = dual_head
        self.norm = norm
        self.eps = eps

        self.depthwise_conv1d = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
        )
        self.conditioning = FiLM1d()

        if nonlinear is not None:
            if nonlinear == "prelu":
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))

            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
            norm_name = "cLN" if causal else "gLN"
            self.norm1d = choose_layer_norm(
                norm_name, in_channels, causal=causal, eps=eps
            )

        if dual_head:
            self.output_pointwise_conv1d = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1
            )

        self.skip_pointwise_conv1d = nn.Conv1d(
            in_channels, skip_channels, kernel_size=1, stride=1
        )

    def forward(
        self, input: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, in_channels, num_samples).
            scale (torch.Tensor):
                Conditioning tensor with shape of (batch_size, in_channels).
            bias (torch.Tensor):
                Conditioning tensor with shape of (batch_size, in_channels).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, in_channels, num_samples).
            torch.Tensor:
                Output tensor for skip connection.
                The shape is (batch_size, in_channels, num_samples).
        """
        nonlinear, norm = self.nonlinear, self.norm
        dual_head = self.dual_head

        x = self.depthwise_conv1d(input)
        x = self.conditioning(x, scale=scale, bias=bias)

        if nonlinear:
            x = self.nonlinear1d(x)

        if norm:
            x = self.norm1d(x)

        if dual_head:
            output = self.output_pointwise_conv1d(x)
        else:
            output = None

        skip = self.skip_pointwise_conv1d(x)

        return output, skip
