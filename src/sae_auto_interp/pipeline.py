import asyncio
from typing import List
from tqdm.asyncio import tqdm
from typing import Callable
from abc import ABC

class Actor(ABC):

    def __init__(
        self, 
        preprocess: Callable = None, 
        postprocess: Callable = None
    ):
        self.preprocess = preprocess
        self.postprocess = postprocess

    async def run(self, input):

        if self.preprocess is not None:
            input = self.preprocess(input)

        result = await self(input)

        if self.postprocess is not None:
            result = self.postprocess(result)

        return result
    
    
class Pipe:
    def __init__(
        self, 
        *actors: List[Actor],
        name: str ="process"
    ):
        self.name = name
        self.actors = actors

    async def run(self, input):
        tasks = [
            actor.run(input) 
            for actor in self.actors
        ]

        return await asyncio.gather(*tasks) 


class Pipeline:
    def __init__(self, generator, *pipes):

        self.generator = generator
        self.pipes = pipes

    async def loop(self, input, pipes):

        if len(pipes) > 0:
            output = await pipes[0].run(input)
            return await self.loop(output, pipes[1:])
        
        return input

    async def run(self, max_processes: int = 200, collate=False):

        sem = asyncio.Semaphore(max_processes)

        async def _process(record):
            async with sem:
                return await self.loop(record, self.pipes)

        for records in self.generator(collate):

            tasks = [
                asyncio.create_task(_process(record)) 
                for record in records
            ]

            for completed_task in tqdm(
                asyncio.as_completed(tasks),
                desc="Collected"
            ):

                await completed_task
                