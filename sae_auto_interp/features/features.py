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
    
    def __repr__(self) -> str:
        return f"layer{self.layer_index}_feature{self.feature_index}"
    
@dataclass
class Example:
    tokens: List[int]
    activations: List[float]
    str_toks: List[str] = None
    max_activation: float = 0.0

    def text(self):
        return "".join(self.str_toks)



class FeatureRecord:

    def __init__(
        self,
        feature: Feature,
        examples: List[Example],
    ):
        self.feature = feature
        self.examples = examples

    @property
    def max_activation(self):
        return self.examples[0].max_activation
    
    def display(
        self, 
        n_examples: int = 5
    ) -> str:
        assert hasattr(self.examples[0], "str_toks"), \
            "Examples must have be detokenized to display."

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
        layer_index: int,
        raw_dir: str,
        tokenizer=None,
        selected_features: List[int] = None,
        processed_dir: str = None, 
        n_random: int = 0,
        min_examples: int = 300,
        max_examples: int = 500
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
        
        # Build location paths
        locations_path = f"{raw_dir}/layer{layer_index}_locations.pt"
        activations_path = f"{raw_dir}/layer{layer_index}_activations.pt"
        
        # Load tensor
        locations = torch.load(locations_path)
        activations = torch.load(activations_path)

        # Get unique features to load into records
        features = torch.unique(locations[:, 2])

        # Filter the features if a selection is provided
        if selected_features is not None:
            selected_features_tensor = torch.tensor(selected_features)
            features = features[torch.isin(features, selected_features_tensor)]
        
        records = []
        for feature_index in tqdm(features, desc=f"Loading features from tensor for layer {layer_index}"):
    
            feature = Feature(
                layer_index=layer_index, 
                feature_index=feature_index.item()
            )

            mask = locations[:, 2] == feature_index
            feature_locations = locations[mask]
            feature_activations = activations[mask]

            record = FeatureRecord.load_record(
                feature,
                tokens,
                feature_locations,
                feature_activations,
                tokenizer=tokenizer,
                processed_dir=processed_dir,
                n_random=n_random,
                min_examples=min_examples,
                max_examples=max_examples,
            )

            if record:
                records.append(record)

        return records
    
    @staticmethod
    def load_record(
        feature: Feature, 
        tokens: Tensor, 
        locations: Tensor,
        activations: Tensor,
        tokenizer = None, 
        processed_dir:str = None, 
        n_random: int = 0,
        min_examples: int = 300,
        max_examples: int = 500
    ):
        """
        Loads a single record from a tensor of locations and activations.
        """
        if len(locations) == 0:
            logger.info(f"{feature} has no activations.")
            return None
        
        if len(locations) < min_examples:
            logger.info(f"{feature} has fewer than {min_examples} activations.")
            return None
        
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
                str_toks=tokenizer.batch_decode(
                    toks, 
                    clean_up_tokenization_spaces=False
                ),
                max_activation=max(acts),
            )
            for toks, acts in zip(
                processed_tokens, 
                processed_activations
            )
        ]

        examples.sort(key=lambda x: x.max_activation, reverse=True)

        record = FeatureRecord(feature, examples)

        # Load processed data if a directory is provided
        if processed_dir:
            record.load_processed(processed_dir)

        return record
    
    def load_processed(self, directory: str):
        path = f"{directory}/layer{self.feature.layer_index}_feature{self.feature.feature_index}.json"
        with bf.BlobFile(path, "rb") as f:
            processed_data = orjson.loads(f.read())
            feature_dict = processed_data.pop("feature")
            self.feature = Feature(**feature_dict)
            self.__dict__.update(processed_data)
    
    def save(self, directory: str, save_examples=False):
        path = f"{directory}/layer{self.feature.layer_index}_feature{self.feature.feature_index}.json"
        serializable = self.__dict__

        if not save_examples:
            serializable.pop("examples")

        with bf.BlobFile(path, "wb") as f:
            f.write(orjson.dumps(serializable))


# These are some terrible implementations from claude we should probably fix.

def get_activating_examples(tokens: torch.Tensor, locations: torch.Tensor, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = tokens.device
    locations = locations.to(device)
    activations = activations.to(device)

    # Create a sparse tensor for activations
    num_sentences, max_sentence_length = tokens.shape
    indices = torch.stack((locations[:, 0].long(), locations[:, 1].long()))
    sparse_activations = torch.sparse_coo_tensor(indices, activations, (num_sentences, max_sentence_length))
    
    # Convert to dense and find active sentences
    dense_activations = sparse_activations.to_dense()
    active_sentences = torch.any(dense_activations != 0, dim=1)
    
    return tokens[active_sentences], dense_activations[active_sentences]


def extract_activation_windows(tokens: torch.Tensor, activations: torch.Tensor, l_ctx: int = CONFIG.l_ctx, r_ctx: int = CONFIG.r_ctx) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    batch_size, max_length = tokens.shape
    device = tokens.device

    # Find the token with max activation for each sentence
    max_activation_indices = torch.argmax(activations, dim=1)

    # Calculate start and end indices for the windows
    start_indices = torch.clamp(max_activation_indices - l_ctx, min=0)
    end_indices = torch.clamp(max_activation_indices + r_ctx + 1, max=max_length)

    # Create masks for valid indices
    row_indices = torch.arange(max_length, device=device).unsqueeze(0)
    masks = (row_indices >= start_indices.unsqueeze(1)) & (row_indices < end_indices.unsqueeze(1))

    # Use masks to get windows
    token_windows = tokens.masked_select(masks).split((end_indices - start_indices).tolist())
    activation_windows = activations.masked_select(masks).split((end_indices - start_indices).tolist())

    return list(token_windows), list(activation_windows)