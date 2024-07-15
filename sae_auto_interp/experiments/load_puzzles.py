from ..features import Example, FeatureRecord
import json

def sentence_to_example(sentence):
    tokens = []
    activations = []
    for part in sentence:
        if type(part) is list:
            tokens.append(part[0])
            activations.append(
                float(part[1])
            )
        else:
            tokens.append(part)
            activations.append(0.0)

    return Example(
        tokens = [], 
        activations = activations,
        str_toks = tokens
    )

def load_puzzles(puzzle_path):
    puzzles = []
    
    with open(puzzle_path) as f:
        data = json.load(f)

    for feature_name, feature_data in data.items():
        feature = FeatureRecord(
            feature = feature_name
        )

        examples = [
            sentence_to_example(sentence) 
            for sentence in feature_data['sentences']
        ]

        feature.examples = examples
        
        puzzles.append({
            "feature": feature,
            "true_explanation": feature_data['explanation'],
            "false_explanations": feature_data['false_explanations']
        })

    return puzzles