from dataclasses import dataclass

from ..features.features import Example, FeatureRecord  
from typing import List
import time
import asyncio

from abc import ABC, abstractmethod

import asyncio
import orjson
import os
import aiofiles
from typing import List, Callable, Awaitable
from datetime import datetime



BATCH_SIZE = 10

@dataclass
class ExplainerInput:
    train_examples: List[Example]
    record: FeatureRecord 

@dataclass
class ExplainerResult:
    explainer_type: str = ""
    input: str = ""
    response: str = ""
    explanation: str = ""

class Explainer(ABC):

    @abstractmethod
    def __call__(
        self,
        explainer_in: ExplainerInput
    ) -> ExplainerResult:
        pass
        

async def run_explainers(
    explainer: Callable[[ExplainerInput], Awaitable[str]],
    explainer_inputs: List[ExplainerInput],
    output_dir: str,
    logging = None,
    batch_size: int = BATCH_SIZE
):
    logger = logging.info if logging else print

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    async def process_and_save(explainer_in, index):
        result = await explainer(explainer_in)
        filename = f"result_{index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(output_dir, filename)
        async with aiofiles.open(filepath, mode='wb') as f:
            await f.write(orjson.dumps(result))
        logger(f"Saved result to {filepath}")
        return result

    async def process_batch(batch, start_index):
        tasks = [process_and_save(explainer_in, i) 
                 for i, explainer_in in enumerate(batch, start=start_index)]
        return await asyncio.gather(*tasks)

    # Split inputs into batches
    explainer_input_batches = [
        explainer_inputs[i:i+batch_size]
        for i in range(0, len(explainer_inputs), batch_size)
    ]

    # Process all batches
    for i, batch in enumerate(explainer_input_batches):
        start_index = i * batch_size
        results = await process_batch(batch, start_index)
        for result in results:
            logger(result)