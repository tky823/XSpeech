from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .film import FiLM1d
from ..modules.utils.tasnet import choose_nonlinear, choose_layer_norm

EPS = 1e-12


class TimeDilatedConvNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 256,
        skip_channels: int = 256,
        kernel_size: int = 3,
        num_blocks: int = 3,
        num_layers: int = 10,
        dilated: bool = True,
        causal: bool = True,
        nonlinear: Optional[str] = None,
        norm: Union[str, bool] = True,
        eps: float = EPS,
    ):
        super().__init__()

        self.num_blocks = num_blocks

        net = []

        for idx in range(num_blocks):
            if idx == num_blocks - 1:
                net.append(
                    TimeDilatedConvBlock1d(
                        num_features,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        num_layers=num_layers,
                        dilated=dilated,
                        causal=causal,
                        nonlinear=nonlinear,
                        norm=norm,
                        dual_head=False,
                        eps=eps,
                    )
                )
            else:
                net.append(
                    TimeDilatedConvBlock1d(
                        num_features,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        num_layers=num_layers,
                        dilated=dilated,
                        causal=causal,
                        nonlinear=nonlinear,
                        norm=norm,
                        dual_head=True,
                        eps=eps,
                    )
                )

        self.net = nn.Sequential(*net)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, num_features, num_samples).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, skip_channels, num_samples).
        """
        num_blocks = self.num_blocks

        x = input
        skip_connection = 0

        for idx in range(num_blocks):
            x, skip = self.net[idx](x)
            skip_connection = skip_connection + skip

        output = skip_connection

        return output


class ConditionedTimeDilatedConvNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 256,
        skip_channels: int = 256,
        kernel_size: int = 3,
        num_blocks: int = 3,
        num_layers: int = 10,
        dilated: bool = True,
        causal: bool = True,
        nonlinear: Optional[str] = None,
        norm: Union[str, bool] = True,
        eps: float = EPS,
    ):
        super().__init__()

        self.num_blocks = num_blocks

        net = []

        for idx in range(num_blocks):
            if idx == num_blocks - 1:
                net.append(
                    TimeDilatedConvBlock1d(
                        num_features,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        num_layers=num_layers,
                        dilated=dilated,
                        causal=causal,
                        nonlinear=nonlinear,
                        norm=norm,
                        dual_head=False,
                        eps=eps,
                    )
                )
            else:
                net.append(
                    TimeDilatedConvBlock1d(
                        num_features,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        num_layers=num_layers,
                        dilated=dilated,
                        causal=causal,
                        nonlinear=nonlinear,
                        norm=norm,
                        dual_head=True,
                        eps=eps,
                    )
                )

        self.net = nn.Sequential(*net)

    def forward(
        self,
        input: torch.Tensor,
        scale: Optional[List[torch.Tensor]] = None,
        bias: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, num_features, num_samples).
            scale (list(torch.Tensor), optional):
                List of input tensor with shape of (batch_size, hidden_channels).
            bias (list(torch.Tensor), optional):
                List of input tensor with shape of (batch_size, hidden_channels).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, skip_channels, num_samples).
        """
        num_blocks = self.num_blocks

        x = input
        skip_connection = 0

        for idx in range(num_blocks):
            if scale is None:
                _scale = None
            else:
                _scale = scale[idx]

            if bias is None:
                _bias = None
            else:
                _bias = bias[idx]

            x, skip = self.net[idx](x, scale=_scale, bias=_bias)
            skip_connection = skip_connection + skip

        output = skip_connection

        return output


class TimeDilatedConvBlock1d(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 256,
        skip_channels: int = 256,
        kernel_size: int = 3,
        num_layers: int = 10,
        dilated: bool = True,
        causal: bool = True,
        nonlinear: Optional[str] = None,
        norm: Union[bool, str] = True,
        dual_head: bool = True,
        eps: float = EPS,
    ):
        super().__init__()

        self.num_layers = num_layers

        net = []

        for idx in range(num_layers):
            if dilated:
                dilation = 2**idx
                stride = 1
            else:
                dilation = 1
                stride = 2

            if not dual_head and idx == num_layers - 1:
                net.append(
                    ResidualBlock1d(
                        num_features,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        causal=causal,
                        nonlinear=nonlinear,
                        norm=norm,
                        dual_head=False,
                        eps=eps,
                    )
                )
            else:
                net.append(
                    ResidualBlock1d(
                        num_features,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        causal=causal,
                        nonlinear=nonlinear,
                        norm=norm,
                        dual_head=True,
                        eps=eps,
                    )
                )

        self.net = nn.Sequential(*net)

    def forward(self, input: torch.Tensor):
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
        num_layers = self.num_layers

        x = input
        skip_connection = 0

        for idx in range(num_layers):
            x, skip = self.net[idx](x)
            skip_connection = skip_connection + skip

        return x, skip_connection


class ConditionedTimeDilatedConvBlock1d(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 256,
        skip_channels: int = 256,
        kernel_size: int = 3,
        num_layers: int = 10,
        dilated: bool = True,
        causal: bool = True,
        nonlinear: Optional[str] = None,
        norm: Union[bool, str] = True,
        dual_head: bool = True,
        eps: float = EPS,
    ):
        super().__init__()

        self.num_layers = num_layers

        net = []

        for idx in range(num_layers):
            if dilated:
                dilation = 2**idx
                stride = 1
            else:
                dilation = 1
                stride = 2

            if not dual_head and idx == num_layers - 1:
                net.append(
                    ConditionedResidualBlock1d(
                        num_features,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        causal=causal,
                        nonlinear=nonlinear,
                        norm=norm,
                        dual_head=False,
                        eps=eps,
                    )
                )
            else:
                net.append(
                    ConditionedResidualBlock1d(
                        num_features,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        causal=causal,
                        nonlinear=nonlinear,
                        norm=norm,
                        dual_head=True,
                        eps=eps,
                    )
                )

        self.net = nn.Sequential(*net)

    def forward(
        self,
        input: torch.Tensor,
        scale: Optional[List[torch.Tensor]] = None,
        bias: Optional[List[torch.Tensor]] = None,
    ):
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, num_features, num_samples).
            scale (list(torch.Tensor), optional):
                List of input tensor with shape of (batch_size, hidden_channels).
            bias (list(torch.Tensor), optional):
                List of input tensor with shape of (batch_size, hidden_channels).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, num_features, num_samples).
            torch.Tensor:
                Output tensor for skip connection.
                The shape is (batch_size, skip_channels, num_samples).
        """
        num_layers = self.num_layers

        x = input
        skip_connection = 0

        for idx in range(num_layers):
            if scale is None:
                _scale = None
            else:
                _scale = scale[idx]

            if bias is None:
                _bias = None
            else:
                _bias = bias[idx]

            x, skip = self.net[idx](x, scale=_scale, bias=_bias)
            skip_connection = skip_connection + skip

        return x, skip_connection


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
            self.nonlinear1d = choose_nonlinear(nonlinear)
            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
            if type(norm) is str:
                norm_name = norm
            elif type(norm) is bool:
                norm_name = "cLN" if causal else "gLN"
            else:
                raise ValueError("Not support {}.".format(norm))

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
            self.nonlinear1d = choose_nonlinear(nonlinear)
            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
            if type(norm) is str:
                norm_name = norm
            elif type(norm) is bool:
                norm_name = "cLN" if causal else "gLN"
            else:
                raise ValueError("Not support {}.".format(norm))

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
        self,
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, num_features, num_samples).
            scale (torch.Tensor, optional):
                Input tensor with shape of (batch_size, num_features).
            bias (torch.Tensor, optional):
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
            self.nonlinear1d = choose_nonlinear(nonlinear)
            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
            if type(norm) is str:
                norm_name = norm
            elif type(norm) is bool:
                norm_name = "cLN" if causal else "gLN"
            else:
                raise ValueError("Not support {}.".format(norm))

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
            self.nonlinear1d = choose_nonlinear(nonlinear)
            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
            if type(norm) is str:
                norm_name = norm
            elif type(norm) is bool:
                norm_name = "cLN" if causal else "gLN"
            else:
                raise ValueError("Not support {}.".format(norm))

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
        self,
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, in_channels, num_samples).
            scale (torch.Tensor, optional):
                Conditioning tensor with shape of (batch_size, in_channels).
            bias (torch.Tensor, optional):
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
