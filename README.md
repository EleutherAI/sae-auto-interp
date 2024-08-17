# Introduction

This is the legacy version of the code used to run the experiments on [our blog post](https://blog.eleuther.ai/autointerp/). We will be using the most recent version for future work.

# Recording feature activations.

Currently, we cache the activations of the features in the SAEs for a given layer given a certain dataset. This is done by running the script `scripts/cache_X.py`. We have different caching scripts for the different experiments we ran. Some of them did not land in the initial post. Caching gpt2 activations happens all at once (because that can fit in an A40), but caching llama activations is done per-layer.


# Generation of explanations.

After caching the feature activations we can generate explanations for the features. We currently use LLama-3 70B quantized to 4-bits as the explanation generator. We run a VLLM server that is called by the script `scripts/explain_X.py`. We have different scripts for the different experiments we ran. We try to have the names be self-explanatory. For most of the scripts the explanations are generated for a simple layer, that can be given as a CLI argument. 


# Score the explanations

To evaluate the explanations, we score them mostly using fuzzing and detection. We have a script `scripts/*score_X.py` that can be used to score the explanations. Most of the scripts have a name associated with the explanation that they are scoring. 
The notable exceptions are `score_all`, that was used to generate the Anthropic style plots, `score_neighbor`, which was used to generate the neighbor plots and `score_simulate`, which was used to generate the simulation plots.

