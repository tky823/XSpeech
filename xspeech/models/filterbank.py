from typing import Optional

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_basis: int,
        kernel_size: int = 16,
        stride: int = 8,
        nonlinear: Optional[str] = None,
    ):
        super().__init__()

        self.kernel_size, self.stride = kernel_size, stride
        self.nonlinear = nonlinear

        self.conv1d = nn.Conv1d(
            in_channels, n_basis, kernel_size=kernel_size, stride=stride, bias=False
        )

        if nonlinear is not None:
            if nonlinear == "relu":
                self.nonlinear1d = nn.ReLU()
            else:
                raise NotImplementedError("Not support {}.".format(nonlinear))

            self.nonlinear = True
        else:
            self.nonlinear = False

    def forward(self, input: torch.Tensor):
        x = self.conv1d(input)

        if self.nonlinear:
            output = self.nonlinear1d(x)
        else:
            output = x

        return output


class Decoder(nn.Module):
    def __init__(
        self, n_basis: int, out_channels: int, kernel_size: int = 16, stride: int = 8
    ):
        super().__init__()

        self.kernel_size, self.stride = kernel_size, stride

        self.conv_transpose1d = nn.ConvTranspose1d(
            n_basis, out_channels, kernel_size=kernel_size, stride=stride, bias=False
        )

    def forward(self, input: torch.Tensor):
        output = self.conv_transpose1d(input)

        return output
