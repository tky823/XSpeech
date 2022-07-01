from typing import Optional

from ..filterbank import Encoder, Decoder


def choose_filterbank(
    hidden_channels: int,
    kernel_size: int,
    stride: Optional[int] = None,
    enc_basis: str = "trainable",
    dec_basis: str = "trainable",
    **kwargs
):
    in_channels = kwargs.get("in_channels") or 1

    if enc_basis == "trainable":
        encoder = Encoder(
            in_channels,
            hidden_channels,
            kernel_size,
            stride=stride,
            nonlinear=kwargs["enc_nonlinear"],
        )
    else:
        raise NotImplementedError("Not support {} for encoder.".format(enc_basis))

    if dec_basis == "trainable":
        decoder = Decoder(hidden_channels, in_channels, kernel_size, stride=stride)
    else:
        raise NotImplementedError("Not support {} for decoder.".format(dec_basis))

    return encoder, decoder
