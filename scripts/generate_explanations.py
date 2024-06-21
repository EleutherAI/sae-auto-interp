import torch
import datasets
from transformer_lens import utils, HookedTransformer
import matplotlib.pyplot as plt
from llama_cpp import Llama
import numpy as np
import time as time
from argparse import ArgumentParser
import json
import tqdm
from sae_auto_interp.features.utils import get_activated_sentences
from sae_auto_interp.explanations.generate import template_explanation,generate_explanation


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--layer",
        type=int,
    )
    parser.add_argument(
        "--model",
        type=str, default="gpt2",
    )
    parser.add_argument(
        "--explainer",
        type=str, default="llama3-70B",
    )

    args = parser.parse_args()
    layer = args.layer
    model_name = args.model
    explainer_name = args.explainer
    #layer = 0
    #model_name = "gpt2"
    #explainer_name = "llama3-70B"

    try:
        features = torch.load(f"/mnt/ssd-1/gpaulo/SAE-Zoology/features/{model_name}/layer_{layer}_features.pt")
        indices = torch.load(f"/mnt/ssd-1/gpaulo/SAE-Zoology/features/{model_name}/layer_{layer}_indices.pt")
    except:
        print("Features and indices not found, please run the feature collection script first.")
        exit()

    ## This is hardcoded for now but I should make a dataset loader
    dataset = datasets.load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False)
    keep_examples = 10000
    dataset = dataset.select(range(keep_examples))


    if model_name != "gpt2":
        print("Model not implemented")
        exit()
    #load the model to get the tokenizer
    model = HookedTransformer.from_pretrained("gpt2", center_writing_weights=False,device="cpu")
    tokenizer = model.tokenizer

    #Tokenize the dataset
    tokenized_data = utils.tokenize_and_concatenate(dataset, tokenizer, max_length=64)
    all_tokens = tokenized_data["tokens"]


    if explainer_name == "llama3-70B":
        explainer = Llama.from_pretrained(
        repo_id="MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF",
        filename="Meta-Llama-3-70B-Instruct.Q8_0-00001-of-00002.gguf",
        n_gpu_layers=-1,
        n_ctx=8192,
        verbose=False
        )
    else:
        print("Invalid explainer")
        exit()

    ## This is pretty hacky for now
    existing_features = torch.unique(indices[:,2])
    explanations_dict = {}
    #n_features = existing_features.shape[0]
    n_features = 100
    for i in tqdm.tqdm(range(n_features)):
        top_sentences, top_activations, top_indices = get_activated_sentences(features, indices, existing_features[i], all_tokens)
        
        #If there are less than 10 sentences, we skip this feature
        if top_sentences.shape[0] < 10:
            continue
        sentences,token_score_pairs = template_explanation(top_sentences,top_activations,top_indices,tokenizer)
        answer = generate_explanation(explainer,sentences,token_score_pairs)
        if answer is None:
            continue
        explanations_dict[int(existing_features[i].item())] = answer

    with open(f"explanations/{model_name}/explanations_layer_{layer}.json","w") as f:
        print(explanations_dict)
        json.dump(explanations_dict,f)
    

