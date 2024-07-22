import asyncio
import time
from tqdm.asyncio import tqdm

MAX_CONCURRENT_TASKS = 10

def generator(n):
    for i in range(n):
        yield [i] * 10

class Pipe:

    def __init__(self, preprocess, *actors):

        self.actors = actors
        self.preprocess = preprocess

    async def _run(self, semaphore, actor, input):

        async with semaphore:
                
            return await actor(input)

    def run(self, input, semaphore):

        input = self.preprocess(input)

        tasks = []

        for actor in self.actors:

            task = asyncio.create_task(
                self._run(semaphore, actor, input)
            )

            tasks.append(task)

        return tasks

class Pipeline:

    def __init__(self, generator, pipes):

        self.generator = generator
        self.pipes = pipes

    async def run(self, max_process: int = 100):

        semaphore = asyncio.Semaphore(max_process)  

        running = []
        total = 0

        for records in self.generator():

            for record in records:
                total += 1

                running.extend(
                    self.pipes[0].run(record, semaphore)
                )

        if len(self.pipes) > 1:

            for pipe in self.pipes[1:]:

                _running = []

                for task in asyncio.as_completed(running):

                    record = await task

                    _running.extend(
                        pipe.run(record, semaphore)
                    )

                running = _running

        results = []

        pbar = tqdm(total=total)
        for completed_task in asyncio.as_completed(running):
            result = await completed_task
            results.append(result)
            pbar.update(1)

        return results
        

                