import torch.nn as nn

from ...modules.norm import GlobalLayerNorm

EPS = 1e-12


def choose_nonlinear(name: str, **kwargs):
    r"""Choose nonlinear function."""
    if name == "prelu":
        module = nn.PReLU()
    elif name == "sigmoid":
        module = nn.Sigmoid()
    elif name == "softmax":
        module = nn.Softmax(**kwargs)
    else:
        raise ValueError("Not support {}.".format(name))

    return module


def choose_layer_norm(
    name: str, num_features: int, causal: bool = False, eps: float = EPS, **kwargs
):
    r"""Choose layer normalization."""
    if name == "gLN":
        if causal:
            raise ValueError("Global Layer Normalization is NOT causal.")
        layer_norm = GlobalLayerNorm(num_features, eps=eps)
    elif name == "BN":
        n_dims = kwargs.get("n_dims") or 1

        if n_dims == 1:
            layer_norm = nn.BatchNorm1d(num_features, eps=eps)
        elif n_dims == 2:
            layer_norm = nn.BatchNorm2d(num_features, eps=eps)
        else:
            raise NotImplementedError(
                "n_dims is expected 1 or 2, but give {}.".format(n_dims)
            )
    else:
        raise NotImplementedError(
            "Not support {} for layer normalization.".format(name)
        )

    return layer_norm
