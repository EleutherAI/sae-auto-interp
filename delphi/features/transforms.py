from typing import Callable, Optional

import torch
from torchtyping import TensorType

from .features import FeatureRecord, prepare_examples
from .loader import BufferOutput

import json
def set_neighbours(
    record: FeatureRecord,
    neighbours_path: str,
    neighbours_type: str,
):
    """
    Set the neighbours for the feature record.
    """
    with open(neighbours_path, "r") as f:
        neighbours = json.load(f)
    record.neighbours = neighbours[neighbours_type][record.feature.feature_index]
