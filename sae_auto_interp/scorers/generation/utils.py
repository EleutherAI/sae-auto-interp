import os
import json
from tqdm import tqdm
import torch

def get_dims(model, examples):
    return model.tokenizer(
        examples, 
        return_tensors='pt',
        padding=True, 
        truncation=True
    ).input_ids.shape

def _score(model, submodule, examples, features):
    batch_size, seq_len = get_dims(model, examples)
    features = torch.tensor(features).long()
    
    with model.trace(examples):
        scores = submodule.ae.output
        expanded_features = features.unsqueeze(1).expand(batch_size, seq_len)
        activations = scores.gather(
            2, expanded_features.unsqueeze(-1)
        ).squeeze(-1)
        scores = torch.max(activations, dim=1).values
        activations.save()

    return activations.value

def to_feature(string):
    string = string.replace(".txt", "").replace("feature", "") 
    return string.split("_")

def score(model, submodule_dict, examples_dir, batch_size=10):

    example_queue = []
    feature_queue = []
    scores = []

    current_submodule = list(submodule_dict.keys())[0]

    for file in tqdm(os.listdir(examples_dir)):
        with open(os.path.join(examples_dir, file), "r") as f:
            examples = json.load(f)['result']

        module, feature = to_feature(file)

        example_queue.append(examples)
        feature_queue.append(int(feature))

        if (
            module != current_submodule 
            or len(example_queue) >= batch_size
        ):
            
            s = _score(
                model, 
                submodule_dict[current_submodule], 
                sum(example_queue, []),
                feature_queue
            )

            scores.append(s)
            example_queue.clear()
            feature_queue.clear()

            current_submodule = module

    return scores