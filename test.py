import asyncio
from functools import wraps

def to_async_generator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        async_generator = AsyncGenerator(func, args, kwargs, loop)
        return async_generator

    return wrapper

class AsyncGenerator:
    def __init__(self, func, args, kwargs, loop):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.loop = loop
        self.iterator = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.iterator is None:
            self.iterator = self.func(*self.args, **self.kwargs)

        try:
            result = await self.loop.run_in_executor(None, next, self.iterator)
            return result
        except 

# Example usage
def sync_generator(n):
    for i in range(n):
        yield i

    return

async def main():
    # Method 1: Using the decorator
    @to_async_generator
    def decorated_sync_generator(n):
        for i in range(n):
            yield i

    async_gen1 = await decorated_sync_generator(5)  # Note the 'await' here
    async for item in async_gen1:
        print(item)

asyncio.run(main())