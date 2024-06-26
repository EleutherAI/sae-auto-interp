# Introduction

This repository is a proof-of-concept that allows for the generation and evaluation of explanations of single features in Sparse Auto-Encoders (SAEs). We currently support gpt-2 SAEs trained by OpenAI [link] and by EleutherAI [link]. 


# Recording feature activations.

Currently, we cache the activations of the features in the SAEs for a given layer given a certain dataset. This is done by running the script `scripts/collect_features_data.py`. We provide a default configuration to collect the activations using the fineweb dataset. The configuration can be found in `sae_auto_interp/features/configs.json`. Collecting features over different datasets should be straightforward by adding a new configuration to the `configs.json` file.

## Example usage

```python scripts/collect_features_data.py --layer 0 --model gpt2 --device cuda:0 --dataset_configuration fineweb_gpt2 --path_autoencoder /path/to/autoencoder ```

# Generation of explanations.

After having collected the features, we can generate explanations for the features. We currently use LLama-3 70B quantized to 4-bits as the explanation generator. Early results show that smaller models are not as capable to generate good explanations. We use the 'llama-cpp' to handle the generation of explanations. To generate the explanations, the model is fed with a certain number of sentences and the tokens that activated the feature (and their corresponding activations, normalized from 0 to 10). The sentences start 20 tokens before the token with the maximum activation and end 4 tokens later (we plan to enable different techniques to select the prompt).
Generating the explanations is as easy as running the script `scripts/generate_explanations.py`. 

## Example usage
    
```python scripts/generate_explanations.py --layer 0 --model gpt2 --number_features 100```

# Score the explanations

Todo.


