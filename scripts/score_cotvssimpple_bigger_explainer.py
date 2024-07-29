import asyncio
from nnsight import LanguageModel
from tqdm import tqdm

from sae_auto_interp.clients import get_client,execute_model
from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_top_and_quantiles
from sae_auto_interp.logger import logger
import random
import argparse
import json

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]

model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
tokens = load_tokenized_data(model.tokenizer,dataset_split="train")

raw_features_path = "raw_features"
processed_features_path = "processed_features"

def load_explanation(explanations_dir,feature):
    explanations_path = f"{explanations_dir}/layer{feature}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    return json.loads(explanation)

scorer_inputs = []
simple_inputs = []
activation_inputs = []
cot_inputs = []
logit_inputs = []
random.seed(22)

for layer in layers:
    module_name = f".transformer.h.{layer}"

    records = FeatureRecord.from_tensor(
        tokens,
        module_name,
        selected_features=list(range(10,50)),
        raw_dir = raw_features_path,
        processed_dir = processed_features_path,
        min_examples=200,
        max_examples=10000
    )

    for record in tqdm(records):
        try:
            simple_explanation = load_explanation("saved_explanations/gpt2_simple",record.feature)["result"]
            activation_explanation = load_explanation("saved_explanations/gpt2_activation",record.feature)["result"]
            logit_explanation = load_explanation("saved_explanations/gpt2_logit",record.feature)["result"]
            cot_explanation = load_explanation("saved_explanations/gpt2_cot",record.feature)["result"]

            record.tokenizer = model.tokenizer
            _, test = sample_top_and_quantiles(
                record=record,
                n_train=0,
                n_test=7,
                n_quantiles=10,
                seed=22,
            )
            non_activating = []
            while len(non_activating)<20:
                
                feature_idx = random.randint(0,len(records)-1)
                if record.feature == records[feature_idx].feature:
                    continue
                idx = random.randint(0,len(records[feature_idx].examples)-1)
                example = records[feature_idx].examples[idx]
                example.decode(model.tokenizer)
                non_activating.append(example)
            extra = []
            for quantiles in test:
                extra.extend(quantiles[-2:])
                quantiles = quantiles[:-2]


        except Exception as e:
            logger.error(f"Failed while sampling for {record.feature}: {e}") 
            continue

        simple_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=simple_explanation,
                extra_examples=extra,
                random_examples=non_activating,
            )
        )
        logit_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=logit_explanation,
                extra_examples=extra,
                random_examples=non_activating,
            )
        )
        activation_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=activation_explanation,
                extra_examples=extra,
                random_examples=non_activating,
            )
        )
        cot_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=cot_explanation,
                extra_examples=extra,
                random_examples=non_activating,
            )
        )
    

#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
#client = get_client("local", "casperhansen/llama-3-8b-instruct-awq", base_url="http://localhost:8002/v1")
client = get_client("openrouter","anthropic/claude-3.5-sonnet",api_key="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")
scorer = FuzzingScorer(client,model.tokenizer,batch_size=5, echo=False,n_few_shots=5)

output_dir = "scores/gpt2_simple_bigger"
asyncio.run(
    execute_model(
        scorer, 
        simple_inputs,
        output_dir=output_dir,
        record_time=True,
        batch_size=1
    )
)

# output_dir = "scores/gpt2_logit"
# asyncio.run(
#     execute_model(
#         scorer, 
#         logit_inputs,
#         output_dir=output_dir,
#         record_time=True
#     )
# )
# output_dir = "scores/gpt2_activation"
# asyncio.run(
#     execute_model(
#         scorer, 
#         activation_inputs,
#         output_dir=output_dir,
#         record_time=True
#     )
# )

# output_dir = "scores/gpt2_cot"
# asyncio.run(
#     execute_model(
#         scorer, 
#         cot_inputs,
#         output_dir=output_dir,
#         record_time=True
#     )
# )