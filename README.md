# Introduction

This library provides utilities for generating and scoring text explanations of sparse autoencoder (SAE) features.

## Installation

Install this library as a local editable installation. Run the following command from the `sae-auto-interp` directory. 

```pip install -e .```

# Loading Autoencoders

This library uses NNsight to load and edit a model with autoencoders.

```python
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

submodule_dict = load_oai_autoencoders(
    model, 
    # List of layers,
    "weights/gpt2_128k",
)
```

# Caching

To cache autoencoder activations, load your autoencoders and run a cache object. 

```python

cache = FeatureCache(
    model, 
    submodule_dict, 
    batch_size = 128,
    filters = module_filter
)

cache.run(n_tokens = 15_000_000, tokens)
```

Caching saves `.safetensors` of `Dict["activations", "locations"]`. 

```python
cache.save_splits(
    n_splits=4, 
    save_dir="/share/u/caden/sae-auto-interp/raw_features/weights"
)
```

Safetensors are split into shards over the width of the autoencoder.

# Loading Feature Records

The `.features` module provides utilities for reconstructing and sampling various statistics for SAE features. 

```python 
from sae_auto_interp.features import FeatureLoader, FeatureDataset

dataset = FeatureDataset(
    raw_dir=raw_features,
    cfg=cfg,
)
```

The feature dataset will construct lazy loaded buffers that load activations into memory when called as an iterator object. You can iterate through the dataset using the `FeatureLoader` object.

```python
loader = FeatureLoader(
    tokens=tokens,
    dataset=dataset,
    constructor = # constructor,
)
```

We use a `max_activation_pooling_sampler` which reconstructs activations given the original cached tokens and each features' locations and activations. It reconstructs a sparse tensor of activations and finds max activating pools.

# Generating Explanations

First, start a VLLM server or your preferred client. Create an explainer from the `.explainers` module. 

```python
SimpleExplainer(
    client, 
    tokenizer = tokenizer, 
)
```

# Scoring Explanations

## Classification

## Simulation

## Generation

# Scripts