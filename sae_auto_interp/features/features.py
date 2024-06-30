from dataclasses import dataclass
from typing import List, Tuple
import torch
from tqdm import tqdm
import orjson
import blobfile as bf
from collections import defaultdict
from typing import List
from ..logger import logger

@dataclass
class Feature:
    layer_index: int
    feature_index: int

    @staticmethod
    def from_dict(layer_feature_dictionary):
        features = []
        for layer, layer_features in layer_feature_dictionary.items():
            features.extend([Feature(int(layer), feature) for feature in layer_features])

        return features
    
@dataclass
class Example:
    tokens: List[int]
    activations: List[float]
    str_toks: List[str]
    text: str
    max_activation: float = 0.0



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
    
    @staticmethod
    def load_record(feature, tokens, tokenizer, feature_dir, processed_dir=None, max_examples=2000):

        path = f"{feature_dir}/layer{feature.layer_index}_feature{feature.feature_index}.json"

        with open(path, "rb") as f:
            locations, activations = orjson.loads(f.read())

        if len(locations) == 0:
            logger.info(f"Feature {feature.feature_index} in layer {feature.layer_index} has no activations.")
            return FeatureRecord(feature, None)
        
        example_tokens, example_activations = get_activating_examples(
            tokens, locations, activations
        )

        processed_tokens, processed_activations = extract_activation_windows(
            example_tokens[:max_examples], 
            example_activations[:max_examples]
        )

        examples = [
            Example(
                tokens=toks,
                activations=acts,
                str_toks=[tokenizer.decode(t) for t in toks],
                text=tokenizer.decode(toks),
                max_activation=max(acts),
            )
            for toks, acts in zip(processed_tokens, processed_activations)
        ]

        examples.sort(key=lambda x: x.max_activation, reverse=True)

        record = FeatureRecord(feature, examples)

        if processed_dir:
            path = f"{processed_dir}/layer{feature.layer_index}_feature{feature.feature_index}.json"
            with bf.BlobFile(path, "rb") as f:
                processed_data = orjson.loads(f.read())
                feature_dict = processed_data.pop("feature")
                record.feature = Feature(**feature_dict)
                record.__dict__.update(processed_data)

        return record
    
    def save(self, directory: str):
        path = f"{directory}/layer{self.feature.layer_index}_feature{self.feature.feature_index}.json"
        serializable = self.__dict__
        serializable.pop("examples")
        with bf.BlobFile(path, "wb") as f:
            f.write(orjson.dumps(serializable))



def sort_features(features):
    layer_sorted_features = defaultdict(list)
    for feature in features:
        layer_sorted_features[feature.layer_index].append(feature)

    return layer_sorted_features


#TODO: We should have a way to load torch tensors 
def feature_loader(
    tokens: List[int],
    features: List,
    model,
    ae_dict,
    feature_dir,
    pipe=False
):
    all_records = []
    
    layer_sorted_features = sort_features(features)

    for layer, features in tqdm(layer_sorted_features.items()):

        records = [
            FeatureRecord.load_record(feature, tokens, model.tokenizer, feature_dir) for feature in features
        ]

        if pipe:
            yield ae_dict[layer], records
        else:
            all_records.append(records)

    return all_records

import torch

def get_activating_examples(tokens: torch.Tensor, locations: torch.Tensor, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets all rows where the feature activates.
    
    Args:
        tokens: List of tokens for each sentence in the batch
        locations: Tensor of feature locations
        activations: Tensor of feature activations
    
    Returns:
        Tensor of activating sentences and their corresponding activations
    """
    # Ensure inputs are tensors
    locations = torch.as_tensor(locations)
    activations = torch.as_tensor(activations)

    # Create a tensor to hold all sentence activations
    max_sentence_length = tokens.shape[1]
    num_sentences = tokens.shape[0]
    all_activations = torch.zeros((num_sentences, max_sentence_length), dtype=torch.float)

    # Use advanced indexing to fill in the activation values
    all_activations[locations[:, 0].long(), locations[:, 1].long()] = activations

    # Find sentences with non-zero activations
    active_sentences = torch.any(all_activations != 0, dim=1)
    
    return tokens[active_sentences], all_activations[active_sentences]

def extract_activation_windows(tokens: torch.Tensor, activations: torch.Tensor, l_ctx: int = 15, r_ctx: int = 4) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Extracts windows around the maximum activation for each sentence.
    
    Args:
        tokens: Tensor of tokens for activating sentences
        activations: Tensor of activations for activating sentences
        l_ctx: Number of tokens to the left
        r_ctx: Number of tokens to the right
    
    Returns:
        List of token windows and activation windows
    """
    # Find the token with max activation for each sentence
    max_activation_indices = torch.argmax(activations, dim=1)

    # Calculate start and end indices for the windows
    start_indices = torch.clamp(max_activation_indices - l_ctx, min=0)
    end_indices = torch.clamp(max_activation_indices + r_ctx + 1, max=tokens.shape[1])

    # Initialize lists to store results
    token_windows = []
    activation_windows = []

    # Extract windows (this part is hard to vectorize due to variable window sizes)
    for i, (start, end) in enumerate(zip(start_indices, end_indices)):
        token_windows.append(tokens[i, start:end])
        activation_windows.append(activations[i, start:end])

    return token_windows, activation_windows
