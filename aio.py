import asyncio
import time
from tqdm.asyncio import tqdm


def generator(n):
    for i in range(n):
        yield [i] * 10

    
class Actor:

    def __init__(self, name):
        self.name = name

    async def _process(self, result):
        await asyncio.sleep(1)
        return result
    
    async def process(self, semaphore, result):
        async with semaphore:
            print(f"Actor {self.name} processing {result}")
            return await self._process(result)

MAX_CONCURRENT_TASKS = 10

async def main():
    start_time = time.time()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    actor_a = Actor("a")
    actor_b = Actor("b")

    actor_a_tasks = []

    for result_list in generator(2):
        for result in result_list:
            task = asyncio.create_task(actor_a.process(semaphore, result))
            actor_a_tasks.append(task)

    actor_b_results = []

    for task in asyncio.as_completed(actor_a_tasks):
        result = await task
        actor_b_task = asyncio.create_task(actor_b.process(semaphore, result))

        actor_b_results.append(actor_b_task)

    results = []
    
    pbar = tqdm(total=20)
    for completed_task in asyncio.as_completed(actor_b_results):
        result = await completed_task
        results.append(result)
        pbar.update(1)

    print("Time taken: ", time.time() - start_time)
    print("Results:", results)

asyncio.run(main())