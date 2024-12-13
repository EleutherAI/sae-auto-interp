# Purpose of this branch

This branch accompanies our blogpost ("Partially rewriting an LLM in natural language")[https://blog.eleuther.ai/generating-text-using-nl-to-simulate-activations/] and or article.

It contains stripped/modified versions of fuzzing and simulation code in the [nl_simulation/lib](nl_simulation/lib) folder, which are used to generate the predictions o the activations.

## Article

We provide both the code to run the experiments as well as data that can be used to reproduce the figures and results in the article. 

In folder [nl_simulation/results](nl_simulation/results/) you can find the scores and explanations for most of the transcoder latents.

In folder [nl_simulation/results/activations/8b](nl_simulation/results/activations/8b) you can find the predicted activations using Llama 8b for most of the transcoder latents, on the last token of 1000 sentences. In that same folder you will find the CE losses from using those predictions to reconstruct activations.

To make the figures in the article you can run the jupyter notebook [nl_simulation/results.ipynb](nl_simulation/results.ipynb).

To fully reproduce the results in the article 4 different steps are needed:

1. Cache the activations for the transcoder latents. This can be done with the script [nl_simulation/cache.py](nl_simulation/cache.py). Check the [config.py](sae_auto_interp/config.py) for the arguments to chose the dataset. The command used in this work was `python cache.py --dataset_repo "monology/pile-uncopyrighted" --dataset_split "train[:1%]" --dataset_name "" --dataset_column_name "text" --batch_size 32 --ctx_len 256 --n_tokens 10000000 --n_splits 5`

2. Generate explanation for each of the latents of the transcoder. This can be done with the script [nl_simulation/generate_explanations.py](nl_simulation/generate_explanations.py). We use Llama 70b to generate these explanations. Llama 70b can be efficiently loaded in 4bit in two a40 gpus. Explaning all 32k latents and scoring them takes 8 a40 gpu days. You can skip this step as we provided the explanations and the scores. The command we used was `python generate_explanations.py --layer 6 --features 32768 --start_feature 0 --experiment_name "transcoder" --width 32768 --n_sentences 1000 --train_type "quantiles" --n_examples_train 40 --n_quantiles 10 --n_random 100 --n_examples_test 100`

3. Run the simulation of the activations. This can be done with the script [nl_simulation/simulation_predictions.py](nl_simulation/simulation_predictions.py). We use Llama 8b to simulate the activations. We use 2 a40 gpus per Llama instance. Simulating all 1000 sentences take 10 a40 gpu days. You can skip this step as we provided the predicted activations. The command we used was `python simulation_predictions.py --num_sentences 1000 --start_sentence 0`

4. Compute the CE loss of using the predictions. This can be done with the script [nl_simulation/ce_reconstruction.py](nl_simulation/ce_reconstruction.py). The command we used was `python ce_reconstruction.py --num_sentences 1000 --start_sentence 0 --fraction 1.0 --random`. The fraction values are -1.0, -2.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0. And the random flag is used to run the random baseline. The fractions -1 and -2 correspond to no active latents and random active latents, respectively.

## Blogpost

The code to run the experiments in the blogpost can be found in the [blog_scripts](blog_scripts) folder.

Code to generate the finetunning data and train the finetuned 8b model can be found in [nl_simulation/finetuning](nl_simulation/finetuning) folder. It assumes you have downloaded the [explanations](https://huggingface.co/datasets/EleutherAI/auto_interp_explanations/blob/main/Gemma/131k/res/model.layers.11_feature.json) and have the cached activations for that layer. 
It can be run with the following commands:
```
python make_data.py --explanation quantiles --top_k 5
python make_data.py --explanation top --top_k -1 --layer_train 11 --layer_test 11

python train.py --dataset quantiles_top5
```

Code to generate the data used in the interactive demo can be found in [nl_simulation/interactive](nl_simulation/interactive) folder.

The script to compute how many latents are correctly identified by the explanations can be found in [blog_scripts/simulation_active.py](blog_scripts/simulation_active.py). It was run with the following commands:
```

python simulation_active.py --model_size 8b --explanation top --window_size 32 --num_sentences 2000 --start_sentence 0 
python simulation_active.py --model_size 8b --explanation quantiles --window_size 32 --num_sentences 2000 --start_sentence 0 
python simulation_active.py --model_size 8b --explanation quantiles --window_size 32 --num_sentences 2000 --start_sentence 0 --score fuzz 
python simulation_active.py --model_size 8b --explanation quantiles --window_size 32 --num_sentences 2000 --start_sentence 0 --score recall 
python simulation_active.py --model_size 8b-top_top5 --explanation top --window_size 32 --num_sentences 2000 --start_sentence 0 
python simulation_active.py --model_size 70b --explanation quantiles --window_size 32 --num_sentences 1000 --start_sentence 0 

```

Some of these comands assume you have finetuned the 8b model, and have access to the [scores](https://huggingface.co/datasets/EleutherAI/auto_interp_explanations/tree/main/scores/gemma/131k/res) of layer 11.


The script to compute the kl divergence if we "cheat" and only care if explanations can correctly identify active latents is in [blog_scripts/kl_div_help.py](blog_scripts/kl_div_help.py). It was run with the following commands:
```
python kl_div_help.py --model_size 8b --explanation top --window_size 32 --num_sentences 2000 --start_sentence 0 
python kl_div_help.py --model_size 8b --explanation quantiles --window_size 32 --num_sentences 2000 --start_sentence 0 
python kl_div_help.py --model_size 8b --explanation quantiles --window_size 32 --num_sentences 2000 --start_sentence 0 --score fuzz 
python kl_div_help.py --model_size 8b --explanation quantiles --window_size 32 --num_sentences 2000 --start_sentence 0 --score recall 
python kl_div_help.py --model_size 8b-top_top5 --explanation top --window_size 32 --num_sentences 2000 --start_sentence 0 
python kl_div_help.py --model_size 70b --explanation quantiles --window_size 32 --num_sentences 1000 --start_sentence 0 

```
