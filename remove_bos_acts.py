from safetensors.numpy import save_file,load_file
#from safetensors.torch import load_file
import torch
import numpy as np
import glob
import time

files = glob.glob("cache/gemma_sae_131k/.*/*.safetensors")
for file in files:
    print(file)
    data = load_file(file)
    locations = data["locations"]
    activations = data["activations"]
    if len(locations) == 0:
        continue
    print(locations[0])
    print(locations[-1])
    new_locations = locations[locations[:,1]!=0]
    new_activations = activations[locations[:,1]!=0]
    new_file = {
        "locations": new_locations,
        "activations": new_activations
    }
    save_file(new_file, file)