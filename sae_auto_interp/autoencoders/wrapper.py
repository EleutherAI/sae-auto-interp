from typing import Callable

import torch


class AutoencoderLatents(torch.nn.Module):
    """
    Wrapper module to simplify capturing of autoencoder latents.
    """

    def __init__(
        self,
        ae: torch.nn.Module,
        _forward: Callable,
        width: int,
    ) -> None:
        super().__init__()
        self.ae = ae
        self._forward = _forward
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)
