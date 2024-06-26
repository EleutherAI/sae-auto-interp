import torch
from transformer_lens import utils, HookedTransformer

from llama_cpp import Llama
import time as time
from argparse import ArgumentParser
import json
import tqdm
from sae_auto_interp.features.dataset import get_all_tokens
from sae_auto_interp.features.utils import get_activated_sentences
from sae_auto_interp.explanations.generate import (
    template_explanation,
    generate_explanation,
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer to collect features from, if -1 will collect from all layers",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices="gpt2,meta-llama/Meta-Llama-3-8B"
    )
    #TODO: Allow for more explainer
    parser.add_argument(
        "--explainer",
        type=str,
        default="llama3-70B",
        choices="llama3-70B",
    )
    parser.add_argument(
        "--number_features",
        type=int,
        default=100,
        help="Number of features to explain, if -1 will explain all features",
    )
    parser.add_argument(
        "--feature_prefix",
        type=str,
        default="",
        help="Prefix used in the feature and index files",
    )
    parser.add_argument(
        "--explanation_prefix",
        type=str,
        default="",
        help="Prefix used in the explanation files",
    )
    parser.add_argument(
        "--max_selection",
        type=int,
        default=100,
        help="Select from the top max_selection sentences",
    )
    parser.add_argument(
        "--number_examples",
        type=int,
        default=10,
        help="Number of examples to use in the template",
    )
    #TODO:Add new sampling tecnique


    args = parser.parse_args()
    layer = args.layer
    model_name = args.model
    explainer_name = args.explainer
    feature_prefix = args.feature_prefix
    explanation_prefix = args.explanation_prefix
    number_features = args.number_features
    max_selection = args.max_selection
    number_examples = args.number_examples

    print("Loading features and indices")
    if "/" in model_name:
        load_name = model_name.split("/")[1]
    else:
        load_name = model_name

    features = torch.load(f"saved_features/{load_name}/{feature_prefix}layer_{layer}_features.pt")
    indices = torch.load(f"saved_features/{load_name}/{feature_prefix}layer_{layer}_indices.pt")
    config = torch.load(f"saved_features/{load_name}/{feature_prefix}layer_{layer}_config.json")

    
    print("Loading model")
    model = HookedTransformer.from_pretrained(
        model_name, center_writing_weights=False, device="cpu"
    )
    tokenizer = model.tokenizer

    all_tokens = get_all_tokens(config,tokenizer)
    
    
    print("Loading explainer")
    explainer = Llama.from_pretrained(
        repo_id="MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF",
        filename="Meta-Llama-3-70B-Instruct.Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=8192,
        verbose=False,
    )

    existing_features = torch.unique(indices[:, 2])
    explanations_dict = {}
    
    if number_features == -1:
        number_features = existing_features.shape[0]
    
    for i in tqdm.tqdm(range(number_features)):
        top_sentences, top_activations, top_indices = get_activated_sentences(
            features, indices, existing_features[i], all_tokens, max_selection
        )

        # If there are less than 10 sentences, we skip this feature
        if top_sentences.shape[0] < 20:
            continue
        sentences, token_score_pairs = template_explanation(
            top_sentences, top_activations, top_indices, tokenizer, number_examples
        )
        # print(sentences,token_score_pairs)
        answer = generate_explanation(explainer, sentences, token_score_pairs)
        if answer is None:
            continue
        explanations_dict[int(existing_features[i].item())] = answer

    with open(
        f"saved_explanations/{load_name}/{explanation_prefix}explanations_layer_{layer}.json", "w"
    ) as f:
        json.dump(explanations_dict, f)
