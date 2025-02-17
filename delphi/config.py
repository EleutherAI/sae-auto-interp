from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Literal

from simple_parsing import Serializable, field, list_field


@dataclass
class ExperimentConfig(Serializable):
    n_examples_train: int = 40
    """Number of examples to sample for latent explanation generation."""

    n_examples_test: int = 50
    """Number of examples to sample for latent explanation testing."""

    n_quantiles: int = 10
    """Number of latent activation quantiles to sample."""

    example_ctx_len: int = 32
    """Length of each sampled example sequence. Longer sequences
    reduce detection scoring performance in weak models."""

    n_non_activating: int = 50
    """Number of non-activating examples to sample."""

    train_type: Literal["top", "random", "quantiles"] = "quantiles"
    """Type of sampler to use for latent explanation generation."""

    test_type: Literal["quantiles", "activation"] = "quantiles"
    """Type of sampler to use for latent explanation testing."""


@dataclass
class LatentConfig(Serializable):
    min_examples: int = 200
    """Minimum number of examples to generate for a single latent.
    If the number of activating examples is less than this, the
    latent will not be explained and scored."""

    max_examples: int = 10_000
    """Maximum number of examples to generate for a single latent."""


@dataclass
class CacheConfig(Serializable):
    dataset_repo: str = "EleutherAI/fineweb-edu-dedup-10b"
    """Dataset repository to use for generating latent activations."""

    faiss_index_type: Literal["flat", "hnsw"] = "hnsw"

    faiss_hnsw_config: dict[str, int] = field(
        default_factory=lambda: {"M": 32, "efConstruction": 200, "efSearch": 128}
    )

    faiss_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    dataset_split: str = "train[:1%]"
    """Dataset split to use for generating latent activations."""

    dataset_name: str = ""
    """Dataset name to use."""

    dataset_column: str = "text"
    """Dataset row to use."""

    batch_size: int = 32
    """Number of sequences to process in a batch."""

    ctx_len: int = 256
    """Context length of the autoencoder. Each batch is shape (batch_size, ctx_len)."""

    n_tokens: int = 10_000_000
    """Number of tokens to cache."""

    n_splits: int = 5
    """Number of splits to divide .safetensors into."""


@dataclass
class RunConfig:
    model: str = field(
        default="meta-llama/Meta-Llama-3-8B",
        positional=True,
    )
    """Name of the model to explain."""

    sparse_model: str = field(
        default="EleutherAI/sae-llama-3-8b-32x",
        positional=True,
    )
    """Name of sparse models associated with the model to explain, or path to
    directory containing their weights. Models must be loadable with sparsify
    or gemmascope."""

    hookpoints: list[str] = list_field()
    """list of model hookpoints to attach sparse models to."""

    explainer_model: str = field(
        default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    )
    """Name of the model to use for explanation and scoring."""

    explainer_model_max_len: int = field(
        default=5120,
    )
    """Maximum length of the explainer model context window."""

    explainer_provider: str = field(
        default="offline",
    )
    """Provider to use for explanation and scoring. Options are 'offline' for local
    models and 'openrouter' for API calls."""

    name: str = ""
    """The name of the run. Results are saved in a directory with this name."""

    max_latents: int | None = None
    """Maximum number of features to explain for each sparse model."""

    filter_bos: bool = False
    """Whether to filter out BOS tokens from the cache."""

    semantic_index: bool = False
    """Whether to build semantic index of token sequences."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    hf_token: str | None = None
    """Huggingface API token for downloading models."""

    pipeline_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""

    num_gpus: int = field(
        default=1,
    )
    """Number of GPUs to use for explanation and scoring."""

    seed: int = field(
        default=22,
    )
    """Seed for the random number generator."""

    verbose: bool = field(
        default=True,
    )
    """Whether to log summary statistics and results of the run."""

    num_examples_per_scorer_prompt: int = field(
        default=5,
    )
    """Number of examples to use for each scorer prompt. Using more than 1 improves
    scoring speed but can leak information to the fuzzing and detection scorer,
    as well as increasing the scorer LLM task difficulty."""

    overwrite: list[Literal["cache", "scores"]] = list_field()
    """List of run stages to recompute. This is a debugging tool
    and may be removed in the future."""
