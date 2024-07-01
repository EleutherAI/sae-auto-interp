from dataclasses import dataclass
from typing import List, Tuple
import torch
from tqdm import tqdm
import orjson
import blobfile as bf
from collections import defaultdict
from typing import List
from ..logger import logger
from torch import Tensor

from .. import example_config as CONFIG

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
    
    def display(
        self, 
        n_examples: int = 5
    ) -> str:
        from IPython.core.display import display, HTML

        def _to_string(tokens, activations):
            result = []
            i = 0
            while i < len(tokens):
                if activations[i] > 0:
                    result.append("<mark>")
                    while i < len(tokens) and activations[i] > 0:
                        result.append(tokens[i])
                        i += 1
                    result.append("</mark>")
                else:
                    result.append(tokens[i])
                    i += 1
            return "".join(result)
        
        strings = [
            _to_string(
                example.str_toks, 
                example.activations
            ) 
            for example in self.examples[:n_examples]
        ]

        display(HTML("<br><br>".join(strings)))
    
    @staticmethod
    def from_tensor(
        tokens: Tensor, 
        tokenizer,
        layer: int,
        raw_dir: str,
        selected_features: List[int] = None,
        processed_dir: str = None, 
        max_examples: int = 2000
    ) -> List["FeatureRecord"]:
        """
        Loads a list of records from a tensor of locations and activations. Pass
        in a proccessed_dir to load a feature's processed data.

        Args:
            tokens: Tokenized data from caching
            tokenizer: Tokenizer from the model
            layer: Layer index
            locations_path: Path to the locations tensor
            activations_path: Path to the activations tensor
            processed_dir: Path to the processed data
            max_examples: Maximum number of examples to load per record

        Returns:
            List of FeatureRecords
        """

        locations_path = f"{raw_dir}/layer{layer}_locations.pt"
        activations_path = f"{raw_dir}/layer{layer}_activations.pt"
        
        locations = torch.load(locations_path)
        activations = torch.load(activations_path)

        features = torch.unique(locations[:, 2])

        records = []
        for feature_index in tqdm(features, desc=f"Loading features from tensor for layer {layer}"):
            if feature_index.item() not in selected_features \
                and selected_features is not None:
                continue
            
            feature = Feature(
                layer_index=layer, 
                feature_index=feature_index.item()
            )

            mask = locations[:, 2] == feature_index
            feature_locations = locations[mask]
            feature_activations = activations[mask]

            record = FeatureRecord.load_record(
                feature,
                tokens,
                tokenizer,
                feature_locations,
                feature_activations,
                processed_dir=processed_dir,
                max_examples=max_examples,
            )

            records.append(record)

        return records
    
    @staticmethod
    def load_record(
        feature, 
        tokens, 
        tokenizer, 
        locations,
        activations,
        processed_dir=None, 
        max_examples=2000
    ):
        if len(locations) == 0:
            logger.info(f"Feature {feature.feature_index} in layer {feature.layer_index} has no activations.")
            return f"{feature.layer_index}_{feature.feature_index} EMPTY"
        
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

def extract_activation_windows(tokens: torch.Tensor, activations: torch.Tensor, l_ctx: int = CONFIG.l_ctx, r_ctx: int = CONFIG.r_ctx) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
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
