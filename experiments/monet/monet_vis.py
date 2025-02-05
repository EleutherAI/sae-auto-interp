# uv run python -m experiments.monet.cache
# uv run python -m examples.example_script --model monet --module .model.layers.12.router --features 20  --width 6143
from matplotlib import pyplot as plt
from glob import glob
import json

import os
score_dir = "results/scores/monet/default/detection"
feature_accs = []
for s in os.listdir(score_dir):
    if not s.endswith(".txt"):
        continue
    data = json.load(open(os.path.join(score_dir, s)))
    corrects = []
    for text in data:
        corrects.append(int(text["correct"]))
    feature_accs.append(sum(corrects)/len(corrects))
plt.hist(feature_accs)
plt.xlabel("Accuracy")
plt.xlim(0, 1)
plt.ylabel("Number of features")
plt.savefig("results/monet_fig.png")