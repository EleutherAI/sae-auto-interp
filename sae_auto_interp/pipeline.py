import asyncio
from functools import wraps
from typing import Callable, List

from tqdm.asyncio import tqdm


def process_wrapper(function, preprocess=None, postprocess=None):
    @wraps(function)
    async def wrapped(input):
        if preprocess is not None:
            input = preprocess(input)

        result = await function(input)

        if postprocess is not None:
            result = postprocess(result)

        return result

    return wrapped


class Pipe:
    def __init__(
        self,
        *functions: List[Callable],
    ):
        self.functions = functions

    async def __call__(self, input):
        tasks = [function(input) for function in self.functions]

        return await asyncio.gather(*tasks)


class Pipeline:
    def __init__(self, generator, *pipes):
        self.generator = generator
        self.pipes = pipes

    async def loop(self, input, pipes):
        if len(pipes) > 0:
            output = await pipes[0](input)
            return await self.loop(output, pipes[1:])

        return input

    async def run(self, max_processes: int = 100, collate=False):
        sem = asyncio.Semaphore(max_processes)

        async def _process(record):
            async with sem:
                return await self.loop(record, self.pipes)

        for records in self.generator(collate):
            tasks = [asyncio.create_task(_process(record)) for record in records]

            for completed_task in tqdm(asyncio.as_completed(tasks), desc="Collected"):
                await completed_task
