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

        return result

start = time.time()
result_ids = [do_some_work.remote(x) for x in range(4)]


results = []

actor = AsyncActor.remote("one")

actor_two = AsyncActor.options(max_concurrency=4).remote("two")

while len(result_ids):

    done_id, result_ids = ray.wait(result_ids)

    result = actor.process_incremental.remote(ray.get(done_id[0]))

    result_two = actor_two.process_incremental.remote(result)

    results.append(result_two)

results = ray.get(results)

print("duration =", time.time() - start, "\nresult = ", results)