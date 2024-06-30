import os
from typing import Dict, List, Tuple
from nnsight import LanguageModel
from sae_auto_interp.features.features import Feature, feature_loader
from sae_auto_interp.features.stats import CombinedStat, Logits
from sae_auto_interp.autoencoders.OpenAI.model import Autoencoder
from sae_auto_interp.autoencoders.model import load_autoencoders
#from keys import openrouter_key
from inspect import iscoroutinefunction
import json
import orjson
from datasets import load_dataset

from transformer_lens import utils

from sae_auto_interp.explainers import (
    Explainer, 
    ExplainerInput, 
    ExplainerResult, 
    Jacob, 
    OpenAIExplainer,
    run_explainers
)
from sae_auto_interp.scorers import (
    DetectionScorer, 
    GenerationScorer, 
    OpenAISimulator, 
    Scorer, 
    ScorerInput, 
    ScorerResult,
    run_scorers
)
from sae_auto_interp.experiments import (
    sample_from_quantiles, sample_top_and_quantiles, sample_top_and_random, sample_top_and_quantiles_single
)

from log import get_logger

logging_path = "./results/logs/run.log"
logger = get_logger("run", logging_path)

model_init_kwargs = {
    "model": "meta-llama/llama-3-70b-instruct",
    "provider": "openrouter",
}


TEST_MODEL_DIR = "openai-community/gpt2"
SIZE = "70b"


def load(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data
    except:
        logger.error(f"Could not load file {path}")
        return False

def save(result, path=None):

    # Check if file exists
    if os.path.exists(path):
        return

    with open(path, "wb") as f:
        json_str = orjson.dumps(result)
        f.write(json_str)    


def load_samples(n_per_layer=50):
    with open("/share/u/caden/scoring/sae_auto_interp/scripts/samples.json", "r") as f:
        samples = json.load(f)

    all_features = []
    for layer, features in samples.items():
        layer = int(layer)
        if layer not in [0,2,4,6,8,10]:
            continue
        features = [
            Feature(
                layer_index=layer,
                feature_index=feature
            ) for feature in features[:n_per_layer]
        ]
        all_features.extend(features)
    
    return all_features

def load_stuff(test_model, ae_dict):
    jacob = Jacob(**model_init_kwargs)

    detection_scorer = DetectionScorer(**model_init_kwargs, test_model_dir=TEST_MODEL_DIR)

    logger.info("Loaded explainers and scorers")

    return [jacob], [detection_scorer]

    
def run(feature_records, test_model, ae_dict):

    explainers, scorers = load_stuff(test_model, ae_dict)

    explainer_results = []
    
    for record in feature_records:
        layer = record.feature.layer_index
        feature = record.feature.feature_index

        # Make directory
        feature_dir = f"./results/layer{layer}_feature{feature}"
        os.makedirs(feature_dir, exist_ok=True)

        try:
            # Load train and test data
            train, test = sample_top_and_quantiles_single(record)[0]
        except:
            logger.error(f"Could not sample data for record {record}")
            continue

        for name, run in run_explainers(
            explainers,
            ExplainerInput(
                train_examples=train,
                record=record,
            ),
            logging=logger
        ):
            path = f"{feature_dir}/{name}_explainer_{SIZE}.json"
            time_path = f"{feature_dir}/{name}_explainer_time_{SIZE}.txt"

            result = load(path)

            if not result:
                runtime, result = run()
                save(result, path=path)
                save(str(runtime), path=time_path)

            explainer_results.append((record, result))

    return

    for record, explainer_result in explainer_results:

        layer = record.feature.layer_index
        feature = record.feature.feature_index
        feature_dir = f"./results/layer{layer}_feature{feature}"

        explainer_type = explainer_result.explainer_type

        scorer_in = ScorerInput(
            explanation=explainer_result.explanation,
            test_examples=test,
            record=record
        )   

        for name, run in run_scorers(
            scorers,
            scorer_in,
            logging=logger
        ):
            path = f"{feature_dir}/{explainer_type}_{name}_scorer_{SIZE}.json"
            time_path = f"{feature_dir}/{explainer_type}_{name}_scorer_time_{SIZE}.txt"

            result = load(path)

            if not result:
                runtime, result = run()
                save(result, path=path)
                save(str(runtime), path=time_path)

            logger.info(f"Scorer {name} result: {result}")


def load_tokens(tokenizer):
    data = load_dataset("stas/openwebtext-10k", split="train")

    tokens = utils.tokenize_and_concatenate(
        data, 
        tokenizer, 
        max_length=64
    )   

    tokens = tokens.shuffle(22)['tokens']
    return tokens

if __name__ == "__main__":

    ae_dict = load_autoencoders("oai", [0, 2, 4, 6, 8, 10])
    logger.info("Loaded autoencoders")

    test_model = LanguageModel(TEST_MODEL_DIR, device_map="cuda:0", dispatch=True)
    test_model.tokenizer.paddding_side = "right"
    logger.info("Loaded test model. Padding set right.")

    features = load_samples(n_per_layer=50)
    tokens = load_tokens(test_model.tokenizer)

    stats = CombinedStat(
        logits = Logits(
            model=test_model,
            top_k_logits=10
        ),
    )    

    all_records = []


    for ae, records in feature_loader(tokens, features, test_model, ae_dict):
        stats.refresh(W_dec=ae.decoder.weight)
        stats.compute(records)
        all_records.extend(records)

    run(
        all_records,
        test_model,
        ae_dict
    )