# %%

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Directory containing the JSON files
directory = '/share/u/caden/sae-auto-interp/scores/cot'

# Function to calculate true positives, false positives, true negatives, and false negatives
def calculate_metrics(data):
    tp = fp = tn = fn = 0
    for item in data:
        if item['activates']:
            if item['marked']:
                tp += 1
            else:
                fn += 1
        else:
            if item['marked']:
                fp += 1
            else:
                tn += 1
    return tp, fp, tn, fn

# Initialize totals
results = defaultdict(dict)

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            data = json.load(file)
            tp, fp, tn, fn = calculate_metrics(data)

            results[filename] = {
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            }


best = sorted(results.items(), key=lambda x: x[1]['tp'], reverse=True)


print(best)          

