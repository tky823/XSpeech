from typing import Optional, List, Union

import torch
import torch.nn as nn

from ..modules.utils.tasnet import choose_nonlinear, choose_layer_norm
from .tdcn import TimeDilatedConvNet, ConditionedTimeDilatedConvNet

EPS = 1e-12


class Separator(nn.Module):
    def __init__(
        self,
        num_features: int,
        bottleneck_channels: int = 128,
        hidden_channels: int = 256,
        skip_channels: int = 128,
        kernel_size: int = 3,
        num_blocks: int = 3,
        num_layers: int = 8,
        dilated: bool = True,
        causal: bool = False,
        nonlinear: Optional[str] = "prelu",
        norm: Union[str, bool] = True,
        mask_nonlinear: str = "sigmoid",
        n_sources: int = 2,
        eps: float = EPS,
    ):
        super().__init__()

        self.num_features, self.n_sources = num_features, n_sources

        if type(norm) is str:
            norm_name = norm
        else:
            norm_name = "cLN" if causal else "gLN"

        self.norm1d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)
        self.bottleneck_conv1d = nn.Conv1d(
            num_features, bottleneck_channels, kernel_size=1, stride=1
        )
        self.tdcn = TimeDilatedConvNet(
            bottleneck_channels,
            hidden_channels=hidden_channels,
            skip_channels=skip_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            num_layers=num_layers,
            dilated=dilated,
            causal=causal,
            nonlinear=nonlinear,
            norm=norm,
        )
        self.prelu = nn.PReLU()
        self.mask_conv1d = nn.Conv1d(
            skip_channels, n_sources * num_features, kernel_size=1, stride=1
        )

        if mask_nonlinear == "sigmoid":
            kwargs = {}
        elif mask_nonlinear == "softmax":
            kwargs = {"dim": 1}

        self.mask_nonlinear = choose_nonlinear(mask_nonlinear, **kwargs)

    def forward(self, input: torch.Tensor):
        r"""Estimate masks for each source.

        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, num_features, num_samples).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, n_sources, num_features, num_samples).
        """
        num_features, n_sources = self.num_features, self.n_sources

        batch_size, _, num_samples = input.size()

        x = self.norm1d(input)
        x = self.bottleneck_conv1d(x)
        x = self.tdcn(x)
        x = self.prelu(x)
        x = self.mask_conv1d(x)
        x = self.mask_nonlinear(x)
        output = x.view(batch_size, n_sources, num_features, num_samples)

        return output


class ConditionedExtractor(nn.Module):
    def __init__(
        self,
        num_features: int,
        bottleneck_channels: int = 128,
        hidden_channels: int = 256,
        skip_channels: int = 128,
        kernel_size: int = 3,
        num_blocks: int = 3,
        num_layers: int = 8,
        dilated: bool = True,
        causal: bool = False,
        nonlinear: Optional[str] = "prelu",
        norm: Union[str, bool] = True,
        mask_nonlinear: str = "sigmoid",
        eps: float = EPS,
    ):
        super().__init__()

        self.num_features = num_features

        if type(norm) is str:
            norm_name = norm
        else:
            norm_name = "cLN" if causal else "gLN"

        self.norm1d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)
        self.bottleneck_conv1d = nn.Conv1d(
            num_features, bottleneck_channels, kernel_size=1, stride=1
        )
        self.tdcn = ConditionedTimeDilatedConvNet(
            bottleneck_channels,
            hidden_channels=hidden_channels,
            skip_channels=skip_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            num_layers=num_layers,
            dilated=dilated,
            causal=causal,
            nonlinear=nonlinear,
            norm=norm,
        )
        self.prelu = nn.PReLU()
        self.mask_conv1d = nn.Conv1d(
            skip_channels, num_features, kernel_size=1, stride=1
        )

        if not mask_nonlinear == "sigmoid":
            raise ValueError(
                "Only support sigmoid function, but given {}.".format(mask_nonlinear)
            )

        self.mask_nonlinear = choose_nonlinear(mask_nonlinear)

    def forward(
        self,
        input: torch.Tensor,
        scale: Optional[List[List[torch.Tensor]]] = None,
        bias: Optional[List[List[torch.Tensor]]] = None,
    ):
        r"""Estimate masks of target source.

        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, num_features, num_samples).
            scale (list(list(torch.Tensor)), optional):
                List of input tensords.
                The shape of each tensor is (batch_size, hidden_channels).
            bias (list(list(torch.Tensor)), optional):
                List of input tensords.
                The shape of each tensor is (batch_size, hidden_channels).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, num_features, num_samples).
        """
        x = self.norm1d(input)
        x = self.bottleneck_conv1d(x)
        x = self.tdcn(x, scale=scale, bias=bias)
        x = self.prelu(x)
        x = self.mask_conv1d(x)
        output = self.mask_nonlinear(x)

        return output
