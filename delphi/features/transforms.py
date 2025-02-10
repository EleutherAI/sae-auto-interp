from typing import Callable, Optional

import torch
from torchtyping import TensorType
from dataclasses import dataclass
from .features import FeatureRecord, prepare_examples
from .loader import BufferOutput

import json


@dataclass
class Neighbour:
    distance: float
    feature_index: int

def set_neighbours(
    record: FeatureRecord,
    neighbours: dict[int, list[tuple[float, int]]],
    threshold: float,
):
    """
    Set the neighbours for the feature record.
    """
    
    neighbours = neighbours[str(record.feature.feature_index)]

    # Each element in neighbours is a tuple of (distance,feature_index)
    # We want to keep only the ones with a distance less than the threshold
    neighbours = [neighbour for neighbour in neighbours if neighbour[0] > threshold]

    record.neighbours = [Neighbour(distance=neighbour[0], feature_index=neighbour[1]) for neighbour in neighbours]
