# %%

import time
import random
import ray
import asyncio

@ray.remote
def do_some_work(x):
    time.sleep(1) # Replace this with work you need to do.
    return x

@ray.remote
class AsyncActor:

    def __init__(self, name):
        self.name = name

    async def process_incremental(self, result):

        print(f"Actor {self.name} processing")
        await asyncio.sleep(1) # Replace this with some processing code.
        print(f"Actor {self.name} processed")

        return result

start = time.time()
result_ids = [do_some_work.remote(x) for x in range(20)]

results = []

actor = AsyncActor.options(max_concurrency=4).remote("one")

actor_two = AsyncActor.options(max_concurrency=4).remote("two")

print(len(result_ids))

while len(result_ids):

    done_id, result_ids = ray.wait(result_ids, num_returns=4)

    result = [actor.process_incremental.remote(i) for i in ray.get(done_id)]

    result_two = [actor_two.process_incremental.remote(i) for i in ray.get(result)]

    results.append(result_two)

r = [ray.get(i)for i in results]

print("duration =", time.time() - start, "\nresult = ", r)