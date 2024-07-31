import json
from collections import OrderedDict, defaultdict

import torch

from ...features.stats import cos
from ...logger import logger


def load_neighbors(all_records, modules, neighbor_file_path):
    with open(neighbor_file_path, "r") as f:
        neighbors = json.load(f)

    to_score = []

    for module_path in modules:
        module_neighbors = neighbors[module_path]

        record_lookup = {}

        # Build record lookup
        for record in all_records:
            feature_index = record.feature.feature_index
            record_lookup[feature_index] = record

        # Loop through features and their neighbors that we want to get
        for feature, feature_neighbors in module_neighbors.items():
            # Set a neighbor attribute on the respective record to store data
            record = record_lookup.get(int(feature), None)

            if record is None:
                logger.info(f"Feature {feature} not found in records")
                continue

            record.neighbors = OrderedDict()

            indices, values = feature_neighbors["indices"], feature_neighbors["values"]

            for index, value in zip(indices, values):
                # Sometimes features are too sparse to have neighbors.
                # Ideally you'd need to cache more tokens in this case.
                try:
                    r = record_lookup[index]
                except KeyError:
                    print(f"Feature {index} not found as neighbor of {feature}")
                    r = None

                record.neighbors[value] = r

            to_score.append(record)

    return to_score


def get_neighbors(submodule_dict, feature_filter, save_dir: str, k=10):
    """
    Get the required features for neighbor scoring.
    """

    neighbors_dict = defaultdict(dict)
    per_layer_features = {}

    for module_path, submodule in submodule_dict.items():
        selected_features = feature_filter.get(module_path, False)

        if not selected_features:
            continue

        W_D = submodule.ae.autoencoder._module.decoder.weight
        cos_sim = cos(W_D, selected_features=selected_features)
        top = torch.topk(cos_sim, k=k)

        top_indices = top.indices
        top_values = top.values

        for i, (indices, values) in enumerate(zip(top_indices, top_values)):
            neighbors_dict[module_path][i] = {
                "indices": indices.tolist()[1:],
                "values": values.tolist()[1:],
            }

        per_layer_features[module_path] = torch.unique(top_indices).tolist()

    with open(f"{save_dir}/neighbors.json", "w") as f:
        json.dump(neighbors_dict, f)

    with open(f"{save_dir}/per_layer_features.json", "w") as f:
        json.dump(per_layer_features, f)
