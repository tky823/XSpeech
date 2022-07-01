import torch
import torch.nn as nn

EPS = 1e-12


class GlobalLayerNorm(nn.Module):
    r"""
    Global layer normalization.
    """

    def __init__(self, num_features: int, eps: float = EPS) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        self.norm = nn.GroupNorm(1, num_features, eps=eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor):
                Input tensor with shape of (batch_size, C, *).

        Returns:
            torch.Tensor:
                Output tensor with shape of (batch_size, C, *).
        """
        output = self.norm(input)

        return output

    def __repr__(self) -> str:
        s = "{}".format(self.__class__.__name__)
        s += "({num_features}, eps={eps})"

        return s.format(**self.__dict__)
