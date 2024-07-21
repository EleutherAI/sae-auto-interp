import time
import ray
import asyncio
import torch


@ray.remote
class Actor:
    def __init__(self, name):
        self.name = name
        self.data = torch.arange(1000)

    def do_some_work(self, index):

        time.sleep(1)
        return self.data[index].item()

@ray.remote
class AsyncActor:
    def __init__(self, name):
        self.name = name

    async def process_incremental(self, result):
        print("processing", result)
        await asyncio.sleep(1) # Replace this with some processing code.
        print("done processing", result)
        return result

ray.init()


start = time.time()

results = []

generator = Actor.options(max_concurrency=4).remote("generator")    


def gen(n):

    for _ in range(n):
        yield [generator.do_some_work.remote(x) for x in range(5)]


actor = AsyncActor.options(max_concurrency=5).remote("one")
actor_two = AsyncActor.options(max_concurrency=5).remote("two")


for result_ids in gen(2):

    done_id, result_ids = ray.wait(result_ids, num_returns=5)

    result = [actor.process_incremental.remote(i) for i in ray.get(done_id)]

    result_two = [actor_two.process_incremental.remote(i) for i in ray.get(result)]

    results.append(result_two)

r = [ray.get(i) for i in results]

print("duration =", time.time() - start, "\nresult = ", r)
