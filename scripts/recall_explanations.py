import torch
import datasets
from transformer_lens import utils, HookedTransformer
from llama_cpp import Llama
import numpy as np
import time as time
from argparse import ArgumentParser
import json
import tqdm

from sae_auto_interp.features.utils import get_activated_sentences
from sae_auto_interp.explanations.generate import template_explanation
from sae_auto_interp.evaluations.choose.utils import recall_evaluations


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
        "--simulator",
        type=str, default="llama3-70B",
    )

    args = parser.parse_args()
    layer = args.layer
    model_name = args.model
    simulator_name = args.simulator

    try:
        features = torch.load(f"/mnt/ssd-1/gpaulo/SAE-Zoology/features/{model_name}/layer_{layer}_features.pt")
        indices = torch.load(f"/mnt/ssd-1/gpaulo/SAE-Zoology/features/{model_name}/layer_{layer}_indices.pt")
    except:
        print("Features and indices not found, please run the feature collection script first.")
        exit()

    try:
        with open(f"explanations/{model_name}/explanations_layer_{layer}.json") as f:
            explanations_dict = json.load(f)
    except:
        print("Explanations not found, please generate the explanations first.")
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


    if simulator_name == "llama3-70B":
        simulator = Llama.from_pretrained(
        repo_id="MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF",
        filename="Meta-Llama-3-70B-Instruct.Q8_0-00001-of-00002.gguf",
        n_gpu_layers=-1,
        n_ctx=8192,
        verbose=False
        )
    else:
        print("Invalid explainer")
        exit()

    explained_features = explanations_dict.keys()
    evaluated_dict = {}
    existing_features = torch.unique(indices[:,2])
    for feature in tqdm.tqdm(explained_features):

        #TODO: Fix this
        feature = int(feature)

        top_sentences, top_activations, top_indices = get_activated_sentences(features, indices, feature, all_tokens)

        # If there are less than 10 sentences, skip
        if top_sentences.shape[0] < 10:
            continue

        sentences,_ = template_explanation(top_sentences,top_activations,top_indices,tokenizer,number_examples=10)

        # Now get 5 random sentences that dont have the feature
        for i in range(10):
            random_feature = np.random.choice(existing_features)
            sentence,_,_ = get_activated_sentences(features, indices, random_feature, all_tokens,max_selection=1)
            sentences.append(sentence)

        explanation = explanations_dict[str(feature)]
        score = recall_evaluations(simulator,sentences,explanation)

        evaluated_dict[int(feature)] = {"explanation":explanation,"number_questions":len(sentences),"correct":score}
        
    with open(f"evaluations/recal/layer_{layer}.json","w") as f:
        json.dump(evaluated_dict,f)




