import torch
from typing import Callable

class AutoencoderLatents(torch.nn.Module):
    """
    Wrapper module to simplify capturing of autoencoder latents.
    """

    def __init__(
        self,
        autoencoder, 
        _forward: Callable,
        n_features: int = 32768,
    ) -> None:
        super().__init__()
        self._forward = _forward
        self.n_features = n_features
        self.autoencoder = autoencoder
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)