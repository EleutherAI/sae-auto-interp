from .latents import LatentRecord, Neighbour


def set_neighbours(
    record: LatentRecord,
    neighbours: dict[str, dict[str, list[tuple[float, int]]]],
    threshold: float,
):
    """
    Set the neighbours for the latent record.
    Neighbours should be a dictionary with module names as keys,
    where the values are a dictionary of latent indices as keys,
    and a list of tuples of (distance,feature_index) as values.
    """

    latent_neighbours = neighbours[record.latent.module_name][
        str(record.latent.latent_index)
    ]

    # Each element in neighbours is a tuple of (distance,feature_index)
    # We want to keep only the ones with a distance less than the threshold
    latent_neighbours = [
        neighbour for neighbour in latent_neighbours if neighbour[0] > threshold
    ]

    record.neighbours = [
        Neighbour(distance=neighbour[0], latent_index=neighbour[1])
        for neighbour in latent_neighbours
    ]
