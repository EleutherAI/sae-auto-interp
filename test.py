# %%
import asyncio
import aiohttp
import random

async def vllm_request(session, endpoint, data):
    await asyncio.sleep(random.randint(0,10))  # Simulate network latency
    return f"Alpha"

async def process_item(session, item):
    # Stage 1
    result1 = await vllm_request(session, 'https://vllm-endpoint1.example.com', {'input': item})

    print("here")
    
    # Stage 2
    result2 = await vllm_request(session, 'https://vllm-endpoint2.example.com', {'input': result1})
    
    print("here2")

    # Stage 3
    final_result = await vllm_request(session, 'https://vllm-endpoint3.example.com', {'input': result2})

    print("here3")
    
    return final_result

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for item in range(10):  # Example: processing 10 items
            tasks.append(asyncio.create_task(process_item(session, item)))
        
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            print(f"Processed result: {result}")

asyncio.run(main())