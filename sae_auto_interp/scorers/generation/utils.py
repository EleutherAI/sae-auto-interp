import json
import os
from typing import Dict, List

import orjson
import torch
from tqdm import tqdm


def score(
    model,
    submodule_dict: Dict,
    examples_dir: str,
    generation_size: int = 10,
    batch_size: int = 10,
):
    counter = 0
    current_module = None
    running_examples = []
    running_features = []

    for file in tqdm(os.listdir(examples_dir)):
        # Load, extract feature information and
        # examples from file
        path = os.path.join(examples_dir, file)
        with open(path, "r") as f:
            examples = json.load(f)
        module, feature = to_feature(file)

        # Can only score one module at a time
        if counter == 0:
            current_module = module

        # Iterate until reach batch size
        if counter < batch_size or current_module == module:
            running_examples.append(examples)
            running_features.append(feature)
            counter += 1
            continue

        # Score batch
        scores = _score(
            model, submodule_dict[module], running_examples, feature, generation_size
        )

        # Save scores
        save(examples, scores, path)

        # Reset
        running_examples = []
        counter = 0


def to_feature(string):
    string = string.replace(".txt", "").replace("feature", "")
    module, feature = string.split("_")
    return module, int(feature)


def _score(
    model,
    submodule: Dict,
    examples: List[List[str]],
    features: List[int],
    generation_size: int,
):
    flattened_examples = sum(examples, [])

    indices = torch.arange(len(flattened_examples))
    splits = indices.split(generation_size)

    all_scores = []

    with model.trace(flattened_examples):
        scores = submodule.ae.output

        for feature, split in zip(features, splits):
            score = torch.any(scores[split, :, feature] != 0, dim=0)

            score = score.sum().item()

            all_scores.append(scores.save())

    return map(lambda x: x.value, all_scores)


def save(examples: List, scores: List[int], path: str):
    for examples, score in zip(examples, scores):
        result = {"examples": examples, "score": score}

        with open(path, "wb") as f:
            f.write(orjson.dumps(result))
