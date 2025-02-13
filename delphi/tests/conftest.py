import pytest
import torch
from transformers import AutoModelForCausalLM

from delphi.config import CacheConfig, RunConfig
from delphi.latents import LatentCache
from delphi.sparse_coders import load_sparse_coders


@pytest.fixture(scope="module")
def cache_setup(tmp_path_factory):
    """
    This fixture creates a temporary directory, loads the model,
    initializes the cache, runs the cache once, saves the cache splits
    and configuration, and returns all the relevant objects.
    """
    # Create a temporary directory for saving cache files and config
    temp_dir = tmp_path_factory.mktemp("test_cache")

    # Load model and set run configuration
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    run_cfg_gemma = RunConfig(
        model="EleutherAI/pythia-70m",
        sparse_model="EleutherAI/sae-pythia-70m-32k",
        hookpoints=["layers.1"],
    )
    hookpoint_to_sae_encode = load_sparse_coders(model, run_cfg_gemma)

    # Define cache config and initialize cache
    cache_cfg = CacheConfig(batch_size=1, ctx_len=256, n_tokens=1000)
    cache = LatentCache(model, hookpoint_to_sae_encode, batch_size=cache_cfg.batch_size)

    # Generate mock tokens and run the cache
    mock_tokens = torch.randint(0, 1000, (4, cache_cfg.ctx_len))
    cache.run(cache_cfg.n_tokens, mock_tokens)

    # Save splits to temporary directory (the layer key is "gpt_neox.layers.1")

    cache.save_splits(n_splits=5, save_dir=temp_dir, save_tokens=True)

    # Save the cache config

    cache.save_config(temp_dir, cache_cfg, "EleutherAI/pythia-70m")

    dict_to_return = {
        "cache": cache,
        "mock_tokens": mock_tokens,
        "cache_cfg": cache_cfg,
        "temp_dir": temp_dir,
    }

    yield dict_to_return
    dict_to_return.clear()
