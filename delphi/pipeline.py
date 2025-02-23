import asyncio
from functools import wraps
from typing import Any, AsyncIterable, Callable

from tqdm.asyncio import tqdm


def process_wrapper(
    function: Callable,
    preprocess: Callable | None = None,
    postprocess: Callable | None = None,
) -> Callable:
    """
    Wraps a function with optional preprocessing and postprocessing steps.

    Args:
        function (Callable): The main function to be wrapped.
        preprocess (Callable, optional): A function to preprocess the input.
            Defaults to None.
        postprocess (Callable, optional): A function to postprocess the output.
            Defaults to None.

    Returns:
        Callable: The wrapped function.
    """

    @wraps(function)
    async def wrapped(input: Any):
        if preprocess is not None:
            input = preprocess(input)

        results = await function(input)

        if postprocess is not None:
            results = postprocess(results)

        return results

    return wrapped


class Pipe:
    """
    Represents a pipe of functions to be executed with the same input.
    """

    def __init__(self, *functions: Callable):
        """
        Initialize the Pipe with a list of functions.

        Args:
            *functions (list[Callable]): Functions to be executed in the pipe.
        """
        self.functions = functions

    async def __call__(self, input: Any) -> list[Any]:
        """
        Execute all functions in the pipe with the given input.

        Args:
            input (Any): The input to be processed by all functions.

        Returns:
            list[Any]: The results of all functions.
        """
        tasks = [function(input) for function in self.functions]

        return await asyncio.gather(*tasks)


class Pipeline:
    """
    Manages the execution of multiple pipes, handling concurrency and progress tracking.
    """

    def __init__(self, loader: AsyncIterable | Callable, *pipes: Pipe | Callable):
        """
        Initialize the Pipeline with a list of pipes.

        Args:
            loader (Callable): The loader to be executed first.
            *pipes (list[Pipe]): Pipes to be executed in the pipeline.
        """
        self.pipes = [loader] + list(pipes)

    async def run(self, max_concurrent: int = 10) -> list[Any]:
        """
        Run the pipeline with a maximum number of concurrent tasks.

        Args:
            max_concurrent: Maximum number of concurrent tasks. Defaults to 10.

        Returns:
            list[Any]: The results of all processed items.
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = set()

        progress_bar = tqdm(desc="Processing items")
        number_of_items = 0

        async def process_and_update(item, semaphore, count):
            result = await self.process_item(item, semaphore, count)
            progress_bar.update(1)
            return result

        async for item in self.generate_items():
            number_of_items += 1
            task = asyncio.create_task(
                process_and_update(item, semaphore, number_of_items)
            )
            tasks.add(task)

            if len(tasks) >= max_concurrent:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                results.extend(task.result() for task in done)
                tasks = pending

        if tasks:
            done, _ = await asyncio.wait(tasks)
            results.extend(task.result() for task in done)

        progress_bar.close()
        return results

    async def generate_items(self) -> AsyncIterable[Any]:
        """
        Generates items from the first pipe, which can be an async iterable or callable

        Yields:
            Any: Items generated from the first pipe.

        Raises:
            TypeError: If the first pipe is neither an async iterable nor a callable.
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

    async def process_item(
        self, item: Any, semaphore: asyncio.Semaphore, count: int
    ) -> Any:
        """
        Processes a single item through all pipes except the first one.

        Args:
            item (Any): The item to be processed.
            semaphore (asyncio.Semaphore): Semaphore for controlling concurrency.
            count (int): The count of the current item being processed.

        Returns:
            Any: The processed item.
        """
        async with semaphore:
            result = item
            for pipe in self.pipes[1:]:
                result = await pipe(result)
        return result
