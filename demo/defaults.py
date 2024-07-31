from torchtyping import TensorType

from sae_auto_interp.features import (
    FeatureRecord,
    pool_max_activation_windows,
    random_activation_windows,
)


def default_constructor(
    record: FeatureRecord,
    tokens: TensorType["batch", "seq"],
    locations: TensorType["locations", 2],
    activations: TensorType["locations"],
    n_random: int,
    ctx_len: int,
    max_examples: int,
):
    pool_max_activation_windows(
        record,
        tokens=tokens,
        locations=locations,
        activations=activations,
        ctx_len=ctx_len,
        max_examples=max_examples,
    )

    random_activation_windows(
        record,
        tokens=tokens,
        locations=locations,
        n_random=n_random,
        ctx_len=ctx_len,
    )
