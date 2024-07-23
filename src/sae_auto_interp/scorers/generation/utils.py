import os
import json
from tqdm import tqdm
import torch
import orjson

def _score(model, submodule, examples, feature):
    
    with model.trace(examples):
        scores = submodule.ae.output
        score = torch.any(
            scores[:, :, feature] != 0, dim=0
        )
        score = score.sum().item()
        score.save()

    return score.value

def to_feature(string):
    string = string.replace(".txt", "").replace("feature", "") 
    return string.split("_")

def score(model, submodule_dict, examples_dir, batch_size=1):


    for file in tqdm(os.listdir(examples_dir)):

        with open(os.path.join(examples_dir, file), "r") as f:
            examples = json.load(f)

        module, feature = to_feature(file)

        score = _score(model, submodule_dict[module], examples, int(feature))

        result = {
            "examples" : examples,
            "score" : score
        }

        with open(os.path.join(examples_dir, file), "wb") as f:
            f.write(orjson.dumps(result))