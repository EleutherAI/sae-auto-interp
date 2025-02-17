from typing import Any

import torch
from transformers import AutoTokenizer

from delphi.config import CacheConfig, ExperimentConfig, LatentConfig
from delphi.latents.loader import LatentDataset


def test_latent_loader(cache_setup: dict[str, Any], tokenizer: AutoTokenizer):
    latent_cfg = LatentConfig(min_examples=0, max_examples=100)
    temp_dir = cache_setup["temp_dir"] 
    print(temp_dir)
    hookpoints = ["layers.1"]
    latent_dict = {
        "layers.1": torch.tensor([0, 7000,14000,21000,28000])
    }
    dataset = LatentDataset(
        raw_dir=str(temp_dir),
        cfg=latent_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
    )