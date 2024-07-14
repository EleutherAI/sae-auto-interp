from .local import Local
from .openrouter import OpenRouter
from .outlines import Outlines
from sae_auto_interp.explainers import SimpleExplainer, ChainOfThought
from sae_auto_interp.scorers import FuzzingScorer
import logging
import os
import orjson 
import asyncio
import aiofiles
import time
import random


def get_client(provider: str, model: str, **kwargs):
    if provider == "local":
        return Local(model=model, **kwargs)
    if provider == "openrouter":
        return OpenRouter(model=model, **kwargs)
    if provider == "outlines":
        return Outlines(model=model, **kwargs)

    return None

async def execute_model(
    model,
    queries,
    output_dir: str,
    record_time=False
):
    """
    Executes a model on a list of queries and saves the results to the output directory.
    """
    from sae_auto_interp.logger import logger

    os.makedirs(output_dir, exist_ok=True)

    async def process_and_save(query):
        layer_index = query.record.feature.layer_index
        feature_index = query.record.feature.feature_index

        logger.info(f"Executing {model.name} on feature layer {layer_index}, feature {feature_index}")

        start_time = time.time()
        result = await model(query)
        end_time = time.time()

        filename = f"layer{layer_index}_feature{feature_index}.txt"
        filepath = os.path.join(output_dir, filename)

        if record_time:
            result = {
                "result": result,
                "time": end_time - start_time
            }
            
        async with aiofiles.open(filepath, mode='wb') as f:
            await f.write(orjson.dumps(result))

        logger.info(f"Saved result to {filepath}")
    
    if isinstance(model, (SimpleExplainer,ChainOfThought)):
        tasks = [process_and_save(query) for query in queries]
        await asyncio.gather(*tasks)
    else:   
        for i in range(0, len(queries)):
            tasks = [process_and_save(queries[i])]
            await asyncio.gather(*tasks)
        
