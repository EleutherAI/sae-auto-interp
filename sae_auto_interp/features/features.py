from dataclasses import dataclass
from typing import List, Tuple
import random

import torch
import orjson
import blobfile as bf

from tqdm import tqdm
from collections import defaultdict
from .. import cache_cfg
from .. import example_cfg
import umap
import logging
from sklearn.neighbors import NearestNeighbors

from typing import List, Optional,Dict
from torch import Tensor


@dataclass
class Feature:
    layer_index: int
    feature_index: int


@dataclass
class Example:
    tokens: List[int]
    activations: List[float]
    str_toks: List[str]
    text: str
    max_activation: float = 0.0


TARGET_NUMBER = 300


class FeatureRecord:

    def __init__(
        self,
        feature: Feature,
        examples: List[Example],
    ):
        self.feature = feature
        self.examples = examples

    def max_activation(self):
        return self.examples[0].max_activation

    def save(self, directory: str):
        path = f"{directory}/layer{self.feature.layer_index}_feature{self.feature.feature_index}.json"
        print(self.__dict__)
        serializable = self.__dict__
        serializable.pop("examples")
        with bf.BlobFile(path, "wb") as f:
            f.write(orjson.dumps(serializable))



def sort_features(features):
    layer_sorted_features = defaultdict(list)
    for feature in features:
        layer_sorted_features[feature.layer_index].append(feature)

    return layer_sorted_features


def load_record(feature, tokens, tokenizer):

    path = (
        f"gs://gpt2-oai-all/layer{feature.layer_index}_feature{feature.feature_index}"
    )

    with bf.BlobFile(path, "rb") as f:
        locations, activations = orjson.loads(f.read())

    example_tokens, example_activations = get_activating_examples(
        tokens, locations, activations, 15, 4
    )

    examples = [
        Example(
            tokens=toks,
            activations=acts,
            str_toks=[tokenizer.decode(t) for t in toks],
            text=tokenizer.decode(toks),
            max_activation=max(acts),
        )
        for toks, acts in zip(example_tokens, example_activations)
    ]

    examples.sort(key=lambda x: x.max_activation, reverse=True)

    return FeatureRecord(feature, examples)

#TODO: We should have a way to load torch tensors 
def feature_loader(
    tokens: List[int],
    features: List,
    model,
    ae_dict,
):
    layer_sorted_features = sort_features(features)
    for layer, features in layer_sorted_features.items():

        records = [
            load_record(feature, tokens, model.tokenizer) for feature in features
        ]

        yield ae_dict[layer], records

#TODO: Should this be somewhere else? Its ok to be here if you think its were it makes more sense but maybe we should have it in some prompt utils.
def prepare_example(example, max_activation=0.0):
    delimited_string = ""
    activation_threshold = max_activation

    pos = 0
    while pos < len(example.tokens):
        if (
            pos + 1 < len(example.tokens)
            and example.activations[pos + 1] > activation_threshold
        ):
            delimited_string += example.str_toks[pos]
            pos += 1
        elif example.activations[pos] > activation_threshold:
            delimited_string += "<<"

            seq = ""
            while (
                pos < len(example.tokens)
                and example.activations[pos] > activation_threshold
            ):

                delimited_string += example.str_toks[pos]
                seq += example.str_toks[pos]
                pos += 1

            delimited_string += ">>"

        else:
            delimited_string += example.str_toks[pos]
            pos += 1

    return delimited_string

# We need this when the features are not saved individually
def select_features(features:Tensor, indices:Tensor, feature_index:Tensor) -> tuple[Tensor,Tensor]:
    return features[indices[:,2] == feature_index], indices[indices[:,2] == feature_index]



def cluster_features(selected_features,selected_indices):
    ## Thanks magic copilot for making this 10x faster
    unique_vals, inverse_indices = torch.unique(selected_indices[:, 0], return_inverse=True)
    counts = torch.bincount(inverse_indices)

    # Sort selected_indices by inverse_indices to group them
    sorted_inverse_indices = inverse_indices.sort()[1]
    sorted_indices = selected_indices[sorted_inverse_indices]
    sorted_features = selected_features[sorted_inverse_indices]
    # Use counts to split sorted_indices into a list of tensors
    count_list = counts.tolist()
    grouped_indices = torch.split(sorted_indices, count_list)
    grouped_features = torch.split(sorted_features, count_list)
    #This aproximates the sum and is faster
    cum_sum = torch.cumsum(sorted_features,dim=0)
    wanted_indices = torch.cumsum(counts,dim=0)
    sum = cum_sum[wanted_indices-1]
    sum = torch.cat((cum_sum[0].unsqueeze(0),sum.diff()))
    return grouped_features,sum,grouped_indices


# The function below is basically the same as this one and I think we should use this one.
def get_activated_sentences(features:Tensor, indices:Tensor, feature_index:Tensor, all_tokens:List[Tensor],max_selection:int=50) -> tuple[Tensor,Tensor,Tensor]:
    selected_features, selected_indices = select_features(features, indices, feature_index)
    selected_features = selected_features
    selected_indices = selected_indices
    sentence_features, sum,sentence_indices = cluster_features(selected_features,selected_indices)
    
    if len(sentence_features) < max_selection:
        max_selection = len(sentence_features)
    else:
        max_selection = max_selection
    if max_selection == -1:
        top_activations = sentence_features
        top_activation_indices = range(len(sentence_features))
        max_selection = len(sentence_features)
    else:
        top_activations = sum.topk(max_selection)
        top_activation_indices = top_activations.indices
        
    all_activations = torch.tensor([])
    all_indices = torch.tensor([])
    sentences = torch.tensor([],dtype=torch.int64)

    #TODO: This look can be slow if max_selection is big, but not sure how to go around it. 
    for counter,idx in enumerate(top_activation_indices):
        t_features = sentence_features[idx]
        t_indices = sentence_indices[idx].clone()
        all_activations = torch.cat((all_activations,t_features))
        sentence = all_tokens[t_indices[0,0].int()]
        sentences = torch.cat((sentences,sentence))
        t_indices[:,0] = counter
        all_indices = torch.cat((all_indices,t_indices))
    sentences = sentences.reshape(max_selection,-1)

    return sentences, all_activations, all_indices

#TODO: #3 This is going to be a bottleneck
# From Claude 3.5!
def get_activating_examples(
    tokens: torch.Tensor, locations, activations, l_ctx: int, r_ctx: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Creates sentences and respective activations given features and locations.

    Args:
        tokens: List of tokens for each sentence in the batch
        locations: Tensor of feature locations
        activations: Tensor of feature activations
        N: Number of tokens to the left
        M: Number of tokens to the right

    Returns:
        List of sentences and activations
    """

    locations = torch.Tensor(locations)
    activations = torch.Tensor(activations)

    # Initialize lists to store results
    sentence_tokens = []
    sentence_activations = []

    # Get unique batch indices
    unique_batches = torch.unique(locations[:, 0])

    for batch_idx in unique_batches:
        # Get all activations for this batch
        batch_mask = locations[:, 0] == batch_idx
        batch_locations = locations[batch_mask]
        batch_activations = activations[batch_mask]

        # Get the sentence tokens
        sentence = tokens[int(batch_idx)]

        # Create activation tensor for the sentence
        sentence_activation = torch.zeros_like(sentence, dtype=torch.float)

        # Fill in the activation values
        for loc, act in zip(batch_locations[:, 1], batch_activations):
            sentence_activation[int(loc)] = act

        # Find the token with max activation
        max_activation_idx = torch.argmax(sentence_activation)

        # Calculate start and end indices for the window
        start_idx = max(0, max_activation_idx - l_ctx)
        end_idx = min(len(sentence), max_activation_idx + r_ctx + 1)

        # Extract the window of tokens and activations
        window_tokens = sentence[start_idx:end_idx]
        window_activations = sentence_activation[start_idx:end_idx]

        # Append to results
        sentence_tokens.append(window_tokens.tolist())
        sentence_activations.append(window_activations.tolist())

    return sentence_tokens, sentence_activations
