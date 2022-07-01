from typing import Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.filterbank import choose_filterbank
from ..modules.utils.tasnet import choose_nonlinear, choose_layer_norm
from .tdcn import TimeDilatedConvNet, ConditionedTimeDilatedConvNet

EPS = 1e-12


class ConvTasNet(nn.Module):
    r"""Conv-TasNet."""

    def __init__(
        self,
        n_basis: int,
        kernel_size: int,
        stride: Optional[int] = None,
        enc_basis: str = "trainable",
        dec_basis: str = "trainable",
        sep_hidden_channels: int = 256,
        sep_bottleneck_channels: int = 128,
        sep_skip_channels: int = 128,
        sep_kernel_size: int = 3,
        sep_num_blocks: int = 3,
        sep_num_layers: int = 8,
        dilated: bool = True,
        sep_nonlinear: str = "prelu",
        sep_norm: Union[str, bool] = True,
        mask_nonlinear: str = "sigmoid",
        causal: bool = True,
        n_sources: int = 2,
        eps: float = EPS,
        **kwargs
    ):
        super().__init__()

        if stride is None:
            stride = kernel_size // 2

        assert kernel_size % stride == 0, "kernel_size is expected divisible by stride"

        # Encoder-decoder
        self.in_channels = kwargs.get("in_channels", 1)
        self.n_basis = n_basis
        self.kernel_size, self.stride = kernel_size, stride
        self.enc_basis, self.dec_basis = enc_basis, dec_basis

        if enc_basis == "trainable":
            self.enc_nonlinear = kwargs["enc_nonlinear"]
        else:
            self.enc_nonlinear = None

        assert (
            enc_basis == "trainable"
        ), "enc_basis should be 'trainable', but given {}.".format(enc_basis)

        # Separator configuration
        self.sep_hidden_channels = sep_hidden_channels
        self.sep_bottleneck_channels = sep_bottleneck_channels
        self.sep_skip_channels = sep_skip_channels
        self.sep_kernel_size = sep_kernel_size
        self.sep_num_blocks = sep_num_blocks
        self.sep_num_layers = sep_num_layers

        self.dilated = dilated
        self.causal = causal
        self.sep_nonlinear = sep_nonlinear
        self.sep_norm = sep_norm
        self.mask_nonlinear = mask_nonlinear

        self.n_sources = n_sources
        self.eps = eps

        # Network configuration
        encoder, decoder = choose_filterbank(
            n_basis,
            kernel_size=kernel_size,
            stride=stride,
            enc_basis=enc_basis,
            dec_basis=dec_basis,
            **kwargs
        )

        self.encoder = encoder
        self.separator = Separator(
            n_basis,
            bottleneck_channels=sep_bottleneck_channels,
            hidden_channels=sep_hidden_channels,
            skip_channels=sep_skip_channels,
            kernel_size=sep_kernel_size,
            num_blocks=sep_num_blocks,
            num_layers=sep_num_layers,
            dilated=dilated,
            causal=causal,
            nonlinear=sep_nonlinear,
            norm=sep_norm,
            mask_nonlinear=mask_nonlinear,
            n_sources=n_sources,
            eps=eps,
        )
        self.decoder = decoder

    def forward(self, input: torch.Tensor):
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, in_channels, num_samples).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, n_sources, num_samples)
                or (batch_size, n_sources, in_channels, num_samples).
        """
        output, _ = self.extract_latent(input)

        return output

    def extract_latent(self, input: torch.Tensor):
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, in_channels, num_samples).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, n_sources, num_samples)
                or (batch_size, n_sources, in_channels, num_samples).
            torch.Tensor:
                Latent tensor with shape of (batch_size, n_sources, n_basis, T),
                where T = (num_samples - kernel_size) // stride + 1.
        """
        n_sources = self.n_sources
        n_basis = self.n_basis
        kernel_size, stride = self.kernel_size, self.stride

        n_dims = input.dim()

        if n_dims == 3:
            batch_size, in_channels, num_samples = input.size()
            assert (
                in_channels == 1
            ), "input.size() is expected (?, 1, ?), but given {}.".format(input.size())
        elif n_dims == 4:
            batch_size, in_channels, n_mics, num_samples = input.size()

            assert (
                in_channels == 1
            ), "input.size() is expected (?, 1, ?, ?), but given {}.".format(
                input.size()
            )

            input = input.view(batch_size, n_mics, num_samples)
        else:
            raise ValueError("Not support {} dimension input.".format(n_dims))

        padding = (stride - (num_samples - kernel_size) % stride) % stride
        padding_left = padding // 2
        padding_right = padding - padding_left

        input = F.pad(input, (padding_left, padding_right))
        w = self.encoder(input)
        mask = self.separator(w)
        w = w.unsqueeze(dim=1)
        w_hat = w * mask

        latent = w_hat
        w_hat = w_hat.view(batch_size * n_sources, n_basis, -1)
        x_hat = self.decoder(w_hat)

        if n_dims == 3:
            x_hat = x_hat.view(batch_size, n_sources, -1)
        else:  # n_dims == 4
            x_hat = x_hat.view(batch_size, n_sources, n_mics, -1)

        output = F.pad(x_hat, (-padding_left, -padding_right))

        return output, latent

    def get_config(self):
        r"""Get dictionary of config.

        Returns:
            dict: Config of model.
        """
        config = {
            "in_channels": self.in_channels,
            "n_basis": self.n_basis,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "enc_basis": self.enc_basis,
            "dec_basis": self.dec_basis,
            "enc_nonlinear": self.enc_nonlinear,
            "sep_hidden_channels": self.sep_hidden_channels,
            "sep_bottleneck_channels": self.sep_bottleneck_channels,
            "sep_skip_channels": self.sep_skip_channels,
            "sep_kernel_size": self.sep_kernel_size,
            "sep_num_blocks": self.sep_num_blocks,
            "sep_num_layers": self.sep_num_layers,
            "dilated": self.dilated,
            "causal": self.causal,
            "sep_nonlinear": self.sep_nonlinear,
            "sep_norm": self.sep_norm,
            "mask_nonlinear": self.mask_nonlinear,
            "n_sources": self.n_sources,
            "eps": self.eps,
        }

        return config

    @property
    def num_parameters(self):
        r"""Compute number of trainable parameters.

        Returns:
            int: Number of parameters.
        """
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters


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
