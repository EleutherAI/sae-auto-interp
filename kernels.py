# %%

import time
import torch.multiprocessing as mp
import torch

def foo(tensor, q):
    mask = tensor == 100
    time.sleep(2)
    q.put(tensor[mask])  # Clone the tensor before putting it in the queue

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    print("spawned")
    
    tensor = torch.arange(10_000)
    tensor.share_memory_() 
    
    processes = []
    results = []

    # Use Manager to create a queue
    with mp.Manager() as manager:
        q = manager.Queue()
        
        for i in range(2):
            p = mp.Process(target=foo, args=(tensor, q))
            p.start()
            processes.append(p)

        # Collect results
        for _ in range(len(processes)):
            results.append(q.get()) 

        # Join processes
        for p in processes:
            p.join()
            # p.join(timeout=5)  # Set a timeout for joining
            # if p.is_alive():
            #     print(f"Process {p.pid} did not terminate, forcefully terminating")
            #     p.terminate()
            #     p.join()

    # Print results
    for result in results:
        print(result)