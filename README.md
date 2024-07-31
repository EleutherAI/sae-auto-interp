# Introduction

This library provides utilities for generating and scoring text explanations of sparse autoencoder (SAE) features. The explainer and scorer models can be run locally or acessed using API calls via OpenRouter.

Note that we're still actively cleaning up the codebase and scripts.

## Installation

Install this library as a local editable installation. Run the following command from the `sae-auto-interp` directory.

```pip install -e .```

# Loading Autoencoders

This library uses NNsight to load and edit a model with autoencoders. One should install version 0.3 of [NNsight](https://github.com/ndif-team/nnsight/tree/0.3)

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

We use a `max_activation_pooling_sampler` which reconstructs activations given the original cached tokens and each features' locations and activations. It reconstructs a sparse tensor of activations and finds maximum activating sets of contexts of a given length.

# Generating Explanations

First, start a VLLM server or your preferred client. Create an explainer from the `.explainers` module.

```python
SimpleExplainer(
    client,
    tokenizer = tokenizer,
)
```

The explainer should be added to a pipe, which will send the explanation requests to the client. The pipe should have a function that happens after the request is completed, to e.g. save the data.

```python
def explainer_postprocess(result):

    with open(f"{explanation_dir}/{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.explanation))

    return result

explainer_pipe = process_wrapper(
    SimpleExplainer(
        client,
        tokenizer=tokenizer,
    ),
    postprocess=explainer_postprocess,
)
```
The pipe should then be used in a pipeline. Running the pipeline will send requests to the client in batches of paralel requests.

```
pipeline = Pipeline(
    loader.load,
    explainer_pipe,
)

asyncio.run(pipeline.run())
```


# Scoring Explanations

The process of running a scorer is similar to that of an explainer. You need to have a client running, and you need to create a Scorer from the '.scorer' module.

```python
RecallScorer(
    client,
    tokenizer=tokenizer,
    batch_size=cfg.batch_size
)
```

You can then create a pipe to run the scorer. The pipe should have a pre-processer, that takes the results from the previous pipe and a post processor, that saves the scores. An scorer should always be run after a explainer pipe, but the explainer pipe can be used to load saved explanations.

```python
def scorer_preprocess(result):
        record = result.record

        record.explanation = result.explanation
        record.extra_examples = record.random_examples

        return record

def scorer_postprocess(result, score_dir):
    with open(f"{score_dir}/{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.score))

scorer_pipe = Pipe(
    process_wrapper(
        RecallScorer(client, tokenizer=tokenizer, batch_size=cfg.batch_size),
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir=recall_dir),
    )
)
```

It is possible to have more than one scorer per pipe. One could use that to run fuzzing and detection together:

```python
scorer_pipe = Pipe(
    process_wrapper(
        RecallScorer(client, tokenizer=tokenizer, batch_size=cfg.batch_size),
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir=recall_dir),
    ),
    process_wrapper(
            FuzzingScorer(client, tokenizer=tokenizer, batch_size=cfg.batch_size),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=fuzz_dir),
        ),
)
```

Then the pipe should be sent to the pipeline and run:

```python
pipeline = Pipeline(
        loader.load,
        explainer_pipe,
        scorer_pipe,
)

asyncio.run(pipeline.run())
```

## Simulation

To do simulation scoring we use a fork of OpenAIs neuron explainer. The same process as described above should be taken but the scorer used should be `OpenAISimulator` our current implementation does not used the LogProbabilities trick, but we are currently working on implementing it such that simulation scoring is less expensive.

## Generation

Generation scoring requires two different passes. One that prompts the model to generate explanations, which uses the same process as the other scorers, and another one that runs the SAEs in the generated sentences and evaluates how many generated examples activate the target feature. An example on how the second step is executed can be found in `demos/generation_score.py`.

# Scripts

Example scripts can be found in `demos`. Some of these scripts can be called from the CLI, as seen in examples found in `scripts`. These baseline scripts should allow anyone to start generating and scoring explanations in any SAE they are interested in. One always needs to first cache the activations of the features of any given SAE, and then generating explanations and scoring them can be done at the same time.

# Experiments

The experiments discussed in [link] were mostly run in a legacy version of this code, which can be found in the [Experiments](https://github.com/EleutherAI/sae-auto-interp/tree/Experiments) branch.


# License

Copyright 2024 the EleutherAI Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
