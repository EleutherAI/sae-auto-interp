import asyncio
import time
from tqdm.asyncio import tqdm

MAX_CONCURRENT_TASKS = 10

def generator(n):
    for i in range(n):
        yield [i] * 10

async def process_incremental(result):
    await asyncio.sleep(1)
    return result

async def worker(semaphore, result):
    async with semaphore:
        return await process_incremental(result)

async def main():
    start_time = time.time()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    tasks = []

    for result_list in generator(2):
        for result in result_list:
            task = asyncio.create_task(worker(semaphore, result))
            tasks.append(task)

    results = []
    with tqdm(total=len(tasks)) as pbar:
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            results.append(result)
            pbar.update(1)

    print("Time taken: ", time.time() - start_time)
    print("Results:", results)

asyncio.run(main())