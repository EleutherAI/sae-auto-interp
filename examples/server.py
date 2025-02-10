from flask import Flask, request, jsonify
import asyncio
import torch
import os
import pandas as pd
from functools import partial
from sentence_transformers import SentenceTransformer
from delphi.clients import OpenRouter
from sklearn.metrics import roc_auc_score


from delphi.explainers import DefaultExplainer
from delphi.features import FeatureRecord, Feature, Example
from delphi.scorers import FuzzingScorer, DetectionScorer,EmbeddingScorer

app = Flask(__name__)

# Global variables
model = None
def calculate_balanced_accuracy(dataframe):
    tp = len(dataframe[(dataframe["ground_truth"]==True) & (dataframe["correct"]==True)])
    tn = len(dataframe[(dataframe["ground_truth"]==False) & (dataframe["correct"]==True)])
    fp = len(dataframe[(dataframe["ground_truth"]==False) & (dataframe["correct"]==False)])
    fn = len(dataframe[(dataframe["ground_truth"]==True) & (dataframe["correct"]==False)])
    if tp+fn == 0:
        recall = 0
    else:
        recall = tp/(tp+fn)
    if tn+fp == 0:
        balanced_accuracy = 0
    else:
        balanced_accuracy = (recall+tn/(tn+fp))/2
    return balanced_accuracy

def per_feature_scores_fuzz_detection(score_data):
    
    data = [d for d in score_data if d.prediction != -1]
    
    data_df = pd.DataFrame(data)
    
    balanced_accuracy = calculate_balanced_accuracy(data_df)
    return balanced_accuracy

def per_feature_scores_embedding(score_data):
    data_df = pd.DataFrame(score_data)
    data_df["ground_truth"] = data_df["distance"]>0
    print(data_df)
    auc_score = roc_auc_score(data_df["ground_truth"],data_df["similarity"])
    return auc_score

def initialize_globals():
    global model
    model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
    
# Initialize globals when the app starts
initialize_globals()

@app.route('/generate_explanation', methods=['POST'])
def generate_explanation():
    """
    Generate an explanation for a given set of activations. This endpoint expects
    a JSON object with the following fields:
    - activations: A list of dictionaries, each containing a 'tokens' key with a list of token strings and a 'values' key with a list of activation values.
    - api_key: The API key to use for the request.
    - model: The model to use for the request.

    We could potentially allow for more options, eg we have a threshold that is set "in stone", we don't do COT and always show activations.
    We don't currently support that, but we could allow for custom prompts as well.  
    """

    data = request.json
    
    if not data or 'activations' not in data:
        return jsonify({"error": "Missing required data"}), 400
    if 'api_key' not in data:
        return jsonify({"error": "Missing API key"}), 400
    if 'model' not in data:
        return jsonify({"error": "Missing model"}), 400
    try:
        feature = Feature(f"feature", 0)
        examples = []
        for activation in data['activations']:
            example = Example(activation['tokens'], torch.tensor(activation['values']))
            examples.append(example)
        feature_record = FeatureRecord(feature)
        feature_record.train = examples
        
        client = OpenRouter(api_key=data['api_key'], model=data['model'])

        explainer = DefaultExplainer(client, tokenizer=None, threshold=0.6)
        result = explainer.call_sync(feature_record)  # Use call_sync instead of __call__

        return jsonify({"explanation": result.explanation}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_score_fuzz_detection', methods=['POST'])
def generate_score_fuzz_detection():
    """
    Generate a score for a given set of activations and explanation. This endpoint expects
    a JSON object with the following fields:
    - activations: A list of dictionaries, each containing a 'tokens' key with a list of token strings and a 'values' key with a list of activation values.
    - explanation: The explanation to use for the score.
    - type: Whether to do detection or fuzzing.
    - api_key: The API key to use for the request.
    - model: The model to use for the request.

    We could potentially allow for more options, eg we hardcode showing 5 examples at a time.
    We don't currently support that, but we could allow for custom prompts as well.  
    OpenRouter doesn't support log_prob, so we can't use that.
    """

    data = request.json
    
    if not data or 'activations' not in data:
        return jsonify({"error": "Missing required data"}), 400
    if 'explanation' not in data:
        return jsonify({"error": "Missing explanation"}), 400
    if 'api_key' not in data:
        return jsonify({"error": "Missing API key"}), 400
    if 'model' not in data:
        return jsonify({"error": "Missing model"}), 400
    if 'type' not in data:
        return jsonify({"error": "Missing type"}), 400
    try:
        feature = Feature(f"feature", 0)
        activating_examples = []
        non_activating_examples = []
        for activation in data['activations']:
            example = Example(activation['tokens'], torch.tensor(activation['values']))
            if sum(activation['values']) > 0:
                activating_examples.append(example)
            else:
                non_activating_examples.append(example)
        feature_record = FeatureRecord(feature)
        feature_record.test = [activating_examples]
        feature_record.extra_examples = non_activating_examples
        feature_record.not_active = non_activating_examples
        feature_record.explanation = data['explanation']
        
        client = OpenRouter(api_key=data['api_key'], model=data['model'])
        if data['type'] == 'fuzz':
            # We can't use log_prob as it's not supported by OpenRouter
            scorer = FuzzingScorer(client, tokenizer=None, n_examples_shown=5,verbose=False,log_prob=False)
        elif data['type'] == 'detection':
            # We can't use log_prob as it's not supported by OpenRouter
            scorer = DetectionScorer(client, tokenizer=None, n_examples_shown=5,verbose=False,log_prob=False)
        result = scorer.call_sync(feature_record)  # Use call_sync instead of __call__
        #print(result.score)
        score = per_feature_scores_fuzz_detection(result.score)
        return jsonify({"score": score,"breakdown": result.score}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_score_embedding', methods=['POST'])
def generate_score_embedding():
    """
    Generate a score for a given set of activations and explanation. This endpoint expects
    a JSON object with the following fields:
    - activations: A list of dictionaries, each containing a 'tokens' key with a list of token strings and a 'values' key with a list of activation values.
    - explanation: The explanation to use for the score.
    """
    global model

    data = request.json
    
    if not data or 'activations' not in data:
        return jsonify({"error": "Missing required data"}), 400
    if 'explanation' not in data:
        return jsonify({"error": "Missing explanation"}), 400
    try:
        feature = Feature(f"feature", 0)
        activating_examples = []
        non_activating_examples = []
        for activation in data['activations']:
            example = Example(activation['tokens'], torch.tensor(activation['values']))
            if sum(activation['values']) > 0:
                activating_examples.append(example)
            else:
                non_activating_examples.append(example)
        feature_record = FeatureRecord(feature)
        feature_record.test = [activating_examples]
        feature_record.extra_examples = non_activating_examples
        feature_record.negative_examples = non_activating_examples
        feature_record.explanation = data['explanation']
        scorer =  EmbeddingScorer(model)
        result = scorer.call_sync(feature_record)  # Use call_sync instead of __call__
        #print(result.score)
        score = per_feature_scores_embedding(result.score)
        return jsonify({"score": score,"breakdown": result.score}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
