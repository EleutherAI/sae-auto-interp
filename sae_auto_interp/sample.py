import random

def get_samples():
    random.seed(22)

    N_LAYERS = 12
    N_FEATURES = 32_768
    N_SAMPLES = 1000

    samples = {}

    for layer in range(N_LAYERS):

        samples[layer] = random.sample(range(N_FEATURES), N_SAMPLES)

    return samples