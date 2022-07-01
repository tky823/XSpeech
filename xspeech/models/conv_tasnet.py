from typing import Optional, Union

import torch
import torch.nn as nn

from ..modules.utils.tasnet import choose_nonlinear, choose_layer_norm
from .tdcn import TimeDilatedConvNet

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
                Input tensor with shape of (batch_size, num_features, n_samples).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, n_sources, n_basis, n_samples).
        """
        num_features, n_sources = self.num_features, self.n_sources

        batch_size, _, n_samples = input.size()

        x = self.norm1d(input)
        x = self.bottleneck_conv1d(x)
        x = self.tdcn(x)
        x = self.prelu(x)
        x = self.mask_conv1d(x)
        x = self.mask_nonlinear(x)
        output = x.view(batch_size, n_sources, num_features, n_samples)

        return output
