from dataclasses import dataclass
from .latents import LatentRecord

@dataclass
class Neighbour:
    distance: float
    feature_index: int

def set_neighbours(
    record: LatentRecord,
    neighbours: dict[int, list[tuple[float, int]]],
    threshold: float,
):
    """
    Set the neighbours for the latent record.
    """
    
    neighbours = neighbours[str(record.latent.latent_index)]

    # Each element in neighbours is a tuple of (distance,feature_index)
    # We want to keep only the ones with a distance less than the threshold
    neighbours = [neighbour for neighbour in neighbours if neighbour[0] > threshold]

    record.neighbours = [Neighbour(distance=neighbour[0], feature_index=neighbour[1]) for neighbour in neighbours]
