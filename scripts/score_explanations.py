import torch
from transformer_lens import HookedTransformer
from llama_cpp import Llama
import numpy as np
import time as time
from argparse import ArgumentParser
import json
import tqdm
from sae_auto_interp.features.dataset import get_all_tokens
from sae_auto_interp.features.utils import get_activated_sentences
from sae_auto_interp.explanations.generate import template_explanation
from sae_auto_interp.evaluations.choose.utils import recall_evaluations
from sae_auto_interp.evaluations.enumerate.utils import enumerate_evaluations,score_evaluations
from sae_auto_interp.autoencoders.model import get_autoencoder


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
    parser.add_argument(
        "--mode",
        type=str, default="recall",
        choices=["recall","generation"],
        help="Choose whether to evaluate recall or generation"
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
        "--number_examples_true",
        type=int,
        default=10,
        help="Number of true examples to use in the template",
    )
    parser.add_argument(
        "--number_examples_false",
        type=int,
        default=10,
        help="Number of false examples to use in the template",
    )

    
    args = parser.parse_args()
    layer = args.layer
    model_name = args.model
    explainer_name = args.explainer
    feature_prefix = args.feature_prefix
    if feature_prefix != "":
        feature_prefix += "_"
    explanation_prefix = args.explanation_prefix
    if explanation_prefix != "":
        explanation_prefix += "_"
    mode = args.mode
    max_selection = args.max_selection
    number_examples_true = args.number_examples_true
    number_examples_false = args.number_examples_false


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
    scorer = Llama.from_pretrained(
        repo_id="MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF",
        filename="Meta-Llama-3-70B-Instruct.Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=8192,
        verbose=False,
    )


    with open(f"saved_explanations/{model_name}/{explanation_prefix}explanations_layer_{layer}.json") as f:
        explanations_dict = json.load(f)
    explained_features = explanations_dict.keys()
    evaluated_dict = {}
    existing_features = torch.unique(indices[:,2])

    if mode == "generation":
        autoencoder = get_autoencoder(model_name,layer,"cuda:0")


    for feature in tqdm.tqdm(explained_features):

        if mode == "recall":
            #TODO: Fix this
            feature = int(feature)
            top_sentences, top_activations, top_indices = get_activated_sentences(features, indices, feature, all_tokens,max_selection)
            if top_sentences.shape[0] < 10:
                continue
            sentences,_ = template_explanation(top_sentences,top_activations,top_indices,tokenizer,number_examples_true)

            for i in range(number_examples_false):
                random_feature = np.random.choice(existing_features)
                sentence,_,_ = get_activated_sentences(features, indices, random_feature, all_tokens,max_selection=1)
                sentences.append(sentence)

            explanation = explanations_dict[str(feature)]
            score = recall_evaluations(scorer,sentences,explanation)
            evaluated_dict[int(feature)] = {"explanation":explanation,"number_questions":len(sentences),"correct":score}
        
        elif mode == "generation":
            score = 0
            feature = int(feature)
            explanation = explanations_dict[str(feature)]
            positive = enumerate_evaluations(scorer,explanation)
            for sentence in positive:
                score += score_evaluations(autoencoder,tokenizer,sentence,model,int(feature),layer,pos=True)
            
            evaluated_dict[int(feature)] = {"explanation":explanation,"number_questions":len(positive),"correct":score}
            
    with open(f"saved_evaluations/recal/{model_name}/human_layer_{layer}.json","w") as f:
        json.dump(evaluated_dict,f)




