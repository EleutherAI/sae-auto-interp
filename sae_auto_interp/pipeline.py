import asyncio
from functools import wraps
from typing import Callable, AsyncIterable

from tqdm.asyncio import tqdm


import time
import cProfile
import io
import pstats

def process_wrapper(function, preprocess=None, postprocess=None):
    """
    Wraps a function with optional preprocessing and postprocessing steps.
    Also adds timing and profiling capabilities.
    """
    @wraps(function)
    async def wrapped(input):
        # Preprocess step
        if preprocess is not None:
            input = preprocess(input)

        ## Profiling setup
        # pr = cProfile.Profile()
        # pr.enable()

        # Timing and function execution
        #start_time = time.time()
        results = await function(input)
        #end_time = time.time()
        #print(f"Finished processing item, {results}")
        # Profiling teardown
        # pr.disable()
        # s = io.StringIO()
        # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        # ps.print_stats()

        # Logging
        #print(f"Function {function.name} took {end_time - start_time:.2f} seconds")
        #print(f"Profiling results for {function.name}:")
        #print(s.getvalue())

        # Postprocess step
        if postprocess is not None:
            results = postprocess(results)
        
        return results

    return wrapped


class Pipe:
    """
    Represents a pipe of functions to be executed with the same imput.
    """
    def __init__(
        self,
        *functions: list[Callable],
    ):
        self.functions = functions
    async def __call__(self, input):
        tasks = [function(input) for function in self.functions]

        return await asyncio.gather(*tasks)
    


class Pipeline:
    """
    Manages the execution of multiple pipes, handling concurrency and progress tracking.
    """
    def __init__(self, *pipes):
        self.pipes = pipes

    async def run(self, max_concurrent: int = 10):
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = set()
        
        progress_bar = tqdm(desc="Processing items")
        number_of_items = 0
        async for item in self.generate_items():
            number_of_items += 1
            task = asyncio.create_task(self.process_item(item, semaphore, number_of_items))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            
            if len(tasks) >= max_concurrent:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                results.extend(task.result() for task in done)
                progress_bar.update(len(done))
                #print(f"Processed {len(done)} items")
        
        if tasks:
            done, _ = await asyncio.wait(tasks)
            results.extend(task.result() for task in done)
            progress_bar.update(len(done))
        
        progress_bar.close()
        return results

    async def generate_items(self):
        """
        Generates items from the first pipe, which can be an async iterable or a callable.
        """
        first_pipe = self.pipes[0]
        
        if isinstance(first_pipe, AsyncIterable):
            async for item in first_pipe:
                yield item
        elif callable(first_pipe):
            for item in first_pipe():
                yield item
                await asyncio.sleep(0)  # Allow other coroutines to run
        else:
            raise TypeError("The first pipe must be an async iterable or a callable")

    async def process_item(self, item, semaphore,count):
        """
        Processes a single item through all pipes except the first one.
        Includes timing for each pipe.
        """
        async with semaphore:
            result = item
            for i, pipe in enumerate(self.pipes[1:], start=1):  # Skip the first pipe (FeatureLoader)
                #start_time = time.time()
                #print(f"Processing item {count} in pipe {i}")
                #if asyncio.iscoroutinefunction(pipe):
                result = await pipe(result)
                #end_time = time.time()
                #print(f"Pipe {i} took {end_time - start_time:.2f} seconds")
            #print(f"Finished processing item {count}")
            return result