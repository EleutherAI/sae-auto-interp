from typing import List, Callable, Awaitable
from .scorers.scorer import ScorerInput
from .explainers import ExplainerInput
import logging
import os
import orjson 
import asyncio
import aiofiles

from transformer_lens import utils
from datasets import load_dataset
from transformers import AutoTokenizer

from ..import cache_config as CONFIG
import random

def get_samples(features_per_layer=None):
    random.seed(22)

    N_LAYERS = 12
    N_FEATURES = 32_768
    N_SAMPLES = 1000

    samples = {}

    for layer in range(N_LAYERS):

        samples[layer] = random.sample(range(N_FEATURES), N_SAMPLES)

    if features_per_layer:
        samples = {
            layer: features[:features_per_layer]
            for layer, features in samples.items()
        }

    return samples

def load_tokenized_data(
    tokenizer: AutoTokenizer
):
    data = load_dataset(CONFIG.dataset_repo, split=CONFIG.dataset_split)

    tokens = utils.tokenize_and_concatenate(
        data, 
        tokenizer, 
        max_length=CONFIG.batch_len
    )   

    tokens = tokens.shuffle(CONFIG.seed)['tokens']

    return tokens

async def execute_model(
    model: Callable[[ScorerInput], Awaitable[str]] | Callable[[ExplainerInput], Awaitable[str]],
    queries: List[ScorerInput] | List[ExplainerInput],
    output_dir: str,
    logging = None
):
    logger = logging.info if logging else print

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    async def process_and_save(query, index):
        result = await model(query)
        layer_index = query.record.feature.layer_index
        feature_index = query.record.feature.feature_index

        filename = f"layer{layer_index}_feature{feature_index}.txt"
        filepath = os.path.join(output_dir, filename)

        async with aiofiles.open(filepath, mode='wb') as f:
            await f.write(orjson.dumps(result))

        logger(f"Saved result to {filepath}")

        return result


    tasks = [process_and_save(query, i) for i, query in enumerate(queries)]
    results = await asyncio.gather(*tasks)

    for result in results:
        logger(result)
