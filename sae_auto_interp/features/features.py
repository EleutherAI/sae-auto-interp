from dataclasses import dataclass
from typing import List, Tuple
import torch
import orjson
import blobfile as bf
from collections import defaultdict
from typing import List


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
    def load_record(feature, tokens, tokenizer, feature_dir, processed_dir=None):

        if processed_dir:
            path = f"{processed_dir}/layer{feature.layer_index}_feature{feature.feature_index}.json"
            with bf.BlobFile(path, "rb") as f:
                processed_data = orjson.loads(f.read())

        path = f"{feature_dir}/layer{feature.layer_index}_feature{feature.feature_index}.json"

        with open(path, "rb") as f:
            locations, activations = orjson.loads(f.read())
        
        example_tokens, example_activations = get_activating_examples(
            tokens, locations, activations
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

        record = FeatureRecord(feature, examples)
        record.__dict__.update(processed_data)

        return record
    
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

    for layer, features in layer_sorted_features.items():

        records = [
            FeatureRecord.load_record(feature, tokens, model.tokenizer, feature_dir) for feature in features
        ]

        if pipe:
            yield ae_dict[layer], records
        else:
            all_records.append(records)

    return all_records

#TODO: #3 This is going to be a bottleneck
# From Claude 3.5!
def get_activating_examples(
    tokens: torch.Tensor, locations, activations, l_ctx: int=15, r_ctx: int=4
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