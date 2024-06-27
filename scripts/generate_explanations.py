import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import time as time
from argparse import ArgumentParser
import json
import tqdm
from sae_auto_interp.features.dataset import get_all_tokens
from sae_auto_interp.features.utils import get_activated_sentences
from sae_auto_interp.clients.local import Local

from sae_auto_interp.explanations.generate import (
    template_explanation,
    get_prompts,
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--layer",
        type=int,
        help="Layer to generate explanations features from",
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
    #This actually does nothing at the moment
    explainer_name = args.explainer

    feature_prefix = args.feature_prefix
    if feature_prefix != "":
        feature_prefix += "_"
    explanation_prefix = args.explanation_prefix
    if explanation_prefix != "":
        explanation_prefix += "_"
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
    #config = torch.load(f"saved_features/{load_name}/{feature_prefix}layer_{layer}_config.json")
    config={
        "dataset_repo": "HuggingFaceFW/fineweb",
        "dataset_name": "sample-10BT",
        "split": "train",
        "number_examples": 50000,
        "max_lenght":256,
        "batch_size": 8
    }
    
    print("Loading model")
    model = HookedTransformer.from_pretrained(
        model_name, center_writing_weights=False, device="cpu"
    )
    tokenizer = model.tokenizer

    all_tokens = get_all_tokens(config,tokenizer)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    
    
    existing_features = torch.unique(indices[:, 2])
    explanations_dict = {}
    
    if number_features == -1:
        number_features = existing_features.shape[0]
    
    prompts=[]

    for i in tqdm.tqdm(range(number_features),desc="Generating prompts"):
        top_sentences, top_activations, top_indices = get_activated_sentences(
            features, indices, existing_features[i], all_tokens, max_selection
        )

        # If there are less than 10 sentences, we skip this feature
        if top_sentences.shape[0] < number_examples:
            continue
        sentences, token_score_pairs = template_explanation(
            top_sentences, top_activations, top_indices, tokenizer, number_examples
        )

        prompt = get_prompts(sentences, token_score_pairs, tokenizer)
        prompts.append(prompt)

    print("Loading explainer")
    explainer_config = {
        "backend": "vllm",
        "model": "casperhansen/llama-3-70b-instruct-awq",
        "quantization": "awq"
    }
    

    explainer = Local(explainer_config["model"],explainer_config)

    if explainer_config["backend"]=="vllm":
        sampling = {"max_tokens": 100}
        # Batch the prompts just to have an estimate of the time
        number_answers = len(prompts)
        batch_size = 10
        answers = []
        pbar = tqdm.tqdm(total=number_answers, desc="Generating explanations")
        for i in range(0, number_answers, batch_size):
            answers += explainer.generate_batch(prompts[i : i + batch_size], sampling)
            pbar.update(batch_size)

    print("Saving explanations")
    for i in range(len(answers)):
        answer = answers[i]
        explanations_dict[int(existing_features[i].item())] = answer

    with open(
        f"saved_explanations/{load_name}/{explanation_prefix}explanations_layer_{layer}.json", "w"
    ) as f:
        json.dump(explanations_dict, f)
