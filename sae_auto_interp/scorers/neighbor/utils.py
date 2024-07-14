import json
from collections import OrderedDict

def load_neighbors(records, all_records, module_path, neighbor_file_path):
    with open(neighbor_file_path, "r") as f:
        neighbors = json.load(f)
        neighbors = neighbors[module_path]

    record_lookup = {}

    # Build record lookup
    for record in all_records:
        feature_index = record.feature.feature_index
        record_lookup[feature_index] = record

    for record in records: 
        record.neighbors = OrderedDict()
        _neighbors = neighbors[str(record.feature.feature_index)]
        indices, values = _neighbors["indices"], _neighbors["values"]
        for index, value in zip(indices, values):
            record.neighbors[value] = record_lookup[index]

