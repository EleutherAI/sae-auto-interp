# Introduction

This library provides utilities for generating and scoring text explanations of sparse autoencoder (SAE) features. The explainer and scorer models can be run locally or acessed using API calls via OpenRouter.

# 

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
