import asyncio
from typing import List
from tqdm.asyncio import tqdm
from typing import Callable

class Actor:

    def __init__(
        self, 
        actor, 
        preprocess: Callable = None, 
        postprocess: Callable = None
    ):
        self.actor = actor
        self.preprocess = preprocess
        self.postprocess = postprocess

    async def _run(self, semaphore, input):

        async with semaphore:
                
            return await self.actor(input)

    def __call__(self, input, semaphore):

        if self.preprocess is not None:
            input = self.preprocess(input)

        task = asyncio.create_task(
            self._run(semaphore, input)
        )

        if self.postprocess is not None:
            task.add_done_callback(
                lambda x: self.postprocess(x)
            )

        return task
    
class Pipe:

    def __init__(
        self, 
        *actors: List, 
        name: str ="process"
    ):
        self.name = name
        self.actors = actors

    def run(self, input, semaphore):

        tasks = []

        for actor in self.actors:
                
            task = actor(input, semaphore)
            tasks.append(task)

        return tasks

class Pipeline:

    def __init__(self, generator, *pipes):

        self.generator = generator
        self.pipes = pipes

    async def run(self, max_processes: int = 1, collate=False):

        sem = asyncio.Semaphore(max_processes)  
        
        for records in self.generator(collate):
            
            for record in records:

                running = []

                running.extend(
                    self.pipes[0].run(record, sem)
                )
            
                for i, pipe in enumerate(self.pipes[1:]):

                    _running = []

                    for task in tqdm(
                        asyncio.as_completed(running), 
                        desc=self.pipes[i].name
                    ):

                        result = await task

                        _running.extend(
                            pipe.run(result, sem)
                        )

                    running = _running

                for completed_task in tqdm(
                    asyncio.as_completed(running), 
                    desc=self.pipes[-1].name
                ):

                    result = await completed_task


                

                