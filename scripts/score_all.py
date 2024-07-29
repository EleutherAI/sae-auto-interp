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

top_10_explainer_inputs = []
top_40_explainer_inputs = []
random_40_explainer_inputs = []
split_top_200_20_all_20_explainer_inputs = []

random.seed(22)

for layer in layers:
    module_name = f".transformer.h.{layer}"

    records = FeatureRecord.from_tensor(
        tokens,
        module_name,
        selected_features=list(range(0,100)),
        raw_dir = raw_features_path,
        processed_dir = processed_features_path,
        min_examples=2000,
        max_examples=100000
    )
    counter = 0
    for record in tqdm(records):
        try:
            top_10_explanation = load_explanation("saved_explanations/gpt2_top_10",record.feature)["result"]
            top_40_explanation = load_explanation("saved_explanations/gpt2_top_40",record.feature)["result"]
            random_40_explanation = load_explanation("saved_explanations/gpt2_random_40",record.feature)["result"]
            split_top_200_20_all_20_explanation = load_explanation("saved_explanations/gpt2_split_top_200_20_all_20",record.feature)["result"]
            record.tokenizer = model.tokenizer
        
            test = [random.sample(record.examples,2000)]
            non_activating = []
            while len(non_activating)<100:
                
                feature_idx = random.randint(0,len(records)-1)
                if record.feature == records[feature_idx].feature:
                    continue
                idx = random.randint(0,len(records[feature_idx].examples)-1)
                example = records[feature_idx].examples[idx]
                example.decode(model.tokenizer)
                non_activating.append(example)
            counter = counter + 1
            if counter > 10:
                break


        except Exception as e:
            logger.error(f"Failed while sampling for {record.feature}: {e}") 
            continue
        
        top_10_explainer_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=top_10_explanation,
                extra_examples=non_activating,
                random_examples=non_activating
            )
        )

        top_40_explainer_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=top_40_explanation,
                extra_examples=non_activating,
                random_examples=non_activating
            )
        )
        
        random_40_explainer_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=random_40_explanation,
                extra_examples=non_activating,
                random_examples=non_activating
            )
        )
        
        split_top_200_20_all_20_explainer_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=split_top_200_20_all_20_explanation,
                extra_examples=non_activating,
                random_examples=non_activating
            )
        )


    


client = get_client("local", "casperhansen/llama-3-70b-instruct-awq", base_url="http://localhost:8002/v1")
scorer = FuzzingScorer(client,model.tokenizer,batch_size=5, echo=True,n_few_shots=5)

# output_dir = "scores/gpt2_top_10_all"
# asyncio.run(
#     execute_model(
#         scorer, 
#         top_10_explainer_inputs,
#         output_dir=output_dir,
#         record_time=True
#     )
# )
# output_dir = "scores/gpt2_top_40_all"
# asyncio.run(
#     execute_model(
#         scorer, 
#         top_40_explainer_inputs,
#         output_dir=output_dir,
#         record_time=True
#     )
# )


output_dir = "scores/gpt2_random_40_all"
asyncio.run(
    execute_model(
        scorer, 
        random_40_explainer_inputs,
        output_dir=output_dir,
        record_time=True
    )
)


output_dir = "scores/gpt2_split_top_200_20_all_20_all"
asyncio.run(
    execute_model(
        scorer, 
        split_top_200_20_all_20_explainer_inputs,
        output_dir=output_dir,
        record_time=True
    )
)
