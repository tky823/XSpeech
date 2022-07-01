import torch


def set_seed(seed: int = 42):
    r"""Set seed for reproductivity."""
    torch.manual_seed(seed)
