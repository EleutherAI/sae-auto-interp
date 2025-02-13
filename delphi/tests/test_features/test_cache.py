import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file
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


def test_latent_locations(cache_setup):
    """
    Test that the latent locations generated in memory have the expected
    shape and values.
    """
    cache = cache_setup["cache"]
    locations = cache.cache.latent_locations["gpt_neox.layers.1"]
    max_values, _ = locations.max(axis=0)
    # Expected values based on the cache run
    assert max_values[0] == 2, "Expected first dimension max value to be 2"
    assert max_values[1] == 255, "Expected token ids to go up to 255"
    assert max_values[2] > 32760, "Expected latent dimension around 32768"


def test_split_files_created(cache_setup):
    """
    Test that exactly 5 cache split files have been created.
    """
    save_dir = cache_setup["temp_dir"] / "gpt_neox.layers.1"
    cache_files = [f for f in os.listdir(save_dir) if f.endswith(".safetensors")]
    assert len(cache_files) == 5, "Expected 5 split files in the cache directory"


def test_split_file_contents(cache_setup):
    """
    Test that one of the split files (loaded via safetensors) holds convincing data:
    - latent locations and activations have the same number of entries,
    - tokens were correctly stored and match the input tokens.
    - latent max values are as expected.
    """
    save_dir = cache_setup["temp_dir"] / "gpt_neox.layers.1"
    mock_tokens = cache_setup["mock_tokens"]
    # Choose one file to verify
    cache_files = os.listdir(save_dir)
    file_path = Path(save_dir) / cache_files[0]
    saved_cache = load_file(str(file_path))

    locations = saved_cache["locations"]
    activations = saved_cache["activations"]
    tokens = saved_cache["tokens"]

    assert len(locations) == len(
        activations
    ), "Mismatch between locations & activations entries"

    np.testing.assert_array_equal(
        tokens,
        mock_tokens[:3, :].cpu().numpy(),
        err_msg="Tokens saved do not match the input tokens",
    )
    max_values = locations.max(axis=0)
    assert max_values[0] == 2, "Max batch index mismatch in saved file"
    assert max_values[1] == 255, "Max token value mismatch in saved file"
    assert max_values[2] > 6520, "Latent dimension mismatch in saved file"


def test_config_file(cache_setup):
    """
    Test that the saved configuration file contains the correct parameters.
    """
    config_path = cache_setup["temp_dir"] / "gpt_neox.layers.1" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    cache_cfg = cache_setup["cache_cfg"]

    assert config["batch_size"] == cache_cfg.batch_size, "Config batch_size mismatch"
    assert config["ctx_len"] == cache_cfg.ctx_len, "Config ctx_len mismatch"
    assert config["n_tokens"] == cache_cfg.n_tokens, "Config n_tokens mismatch"
