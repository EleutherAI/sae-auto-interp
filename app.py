from flask import Flask, request, jsonify
import asyncio
import torch
from functools import partial

from sae_auto_interp.clients import OpenRouter
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.counterfactuals import ExplainerInterventionExample, ExplainerNeuronFormatter, get_explainer_prompt, fs_examples

app = Flask(__name__)

# Global variables
client = None
explainer_pipe = None

def initialize_globals():
    # Make the folder for storing the explanations
    os.makedirs("explanations", exist_ok=True)

    # Make the folder for storing the scores

@app.before_first_request
def before_first_request():
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
            example = Example(activation['tokens'], activation['values'])
            examples.append(example)
        feature_record = FeatureRecord(feature)
        feature_record.train = [examples]
        
        client = OpenRouter(api_key=data['api_key'], model=data['model'])

        explainer = DefaultExplainer(client, tokenizer=None, threshold=0.6)
        explanation = explainer(feature_record).explanation

        return jsonify({"explanation": explanation}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_score_fuzz', methods=['POST'])
def generate_score_fuzz():
    """
    Generate a score for a given set of activations. This endpoint expects
    a JSON object with the following fields:
    - activations: A list of dictionaries, each containing a 'tokens' key with a list of token strings and a 'values' key with a list of activation values.
    """

if __name__ == '__main__':
    app.run(debug=True)
