from typing import Optional

import torch
import torch.nn as nn


class FiLM(nn.Module):
    r"""Feature-wise Linear Modulation"""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, num_features, *).
            scale (torch.Tensor, optional):
                Conditioning tensor with shape of (batch_size, num_features).
            bias (torch.Tensor, optional):
                Conditioning tensor with shape of (batch_size, num_features).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, num_features, *).
        """
        n_dims = input.dim()
        expand_dims = (1,) * (n_dims - 2)
        dims = scale.size() + expand_dims

        if scale is None:
            gamma = 1
        else:
            gamma = scale.view(*dims)

        if bias is None:
            beta = 0
        else:
            beta = bias.view(*dims)

        return gamma * input + beta


class FiLM1d(FiLM):
    r"""Feature-wise Linear Modulation for audio-like input."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, num_features, num_samples).
            scale (torch.Tensor, optional):
                Conditioning tensor with shape of (batch_size, num_features).
            bias (torch.Tensor, optional):
                Conditioning tensor with shape of (batch_size, num_features).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, num_features, num_samples).
        """
        dims = scale.size() + (1,)

        if scale is None:
            gamma = 1
        else:
            gamma = scale.view(*dims)

        if bias is None:
            beta = 0
        else:
            beta = bias.view(*dims)

        return gamma * input + beta
