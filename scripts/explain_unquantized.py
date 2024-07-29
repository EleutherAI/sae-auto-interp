import asyncio
import random

from nnsight import LanguageModel

from sae_auto_interp.explainers import SimpleExplainer, ExplainerInput
from sae_auto_interp.clients import get_client, execute_model
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features import FeatureRecord
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]

model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
tokens = load_tokenized_data(model.tokenizer,dataset_split="train")

raw_features_path = "raw_features"
processed_features_path = "processed_features"
explainer_inputs=[]
random.seed(22)


for layer in layers:
    module_name = f".transformer.h.{layer}"

    records = FeatureRecord.from_tensor(
        tokens,
        module_name,
        selected_features=list(range(0,50)),
        raw_dir = raw_features_path,
        processed_dir = processed_features_path,
        min_examples=200,
        max_examples=10000
    )

    for record in records:

        examples = record.examples
        train_examples = random.sample(examples[:200], 40)
        for example in train_examples:
            normalized_activations = (example.activations / record.max_activation)*10
            example.normalized_activations = normalized_activations.round()
        explainer_inputs.append(
            ExplainerInput(
                train_examples=train_examples,
                record=record
            )
        )

client = get_client("local", "meta-llama/Meta-Llama-3-70B-Instruct", base_url="http://localhost:8002/v1")

simple_explainer = SimpleExplainer(client, tokenizer=model.tokenizer,cot=False,logits=False,activations=False)
explainer_out_dir = "saved_explanations/gpt2_simple_unq"

asyncio.run(
    execute_model(
        simple_explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
        record_time=True
    )
)

# simple_explainer = SimpleExplainer(client, tokenizer=model.tokenizer,cot=False,logits=False,activations=True)
# explainer_out_dir = "saved_explanations/gpt2_activation"

# asyncio.run(
#     execute_model(
#         simple_explainer, 
#         explainer_inputs,
#         output_dir=explainer_out_dir,
#         record_time=True
#     )
# )


# logit_explainer = SimpleExplainer(client, tokenizer=model.tokenizer,cot=False,logits=True,activations=True)
# explainer_out_dir = "saved_explanations/gpt2_logit"

# asyncio.run(
#     execute_model(
#         logit_explainer, 
#         explainer_inputs,
#         output_dir=explainer_out_dir,
#         record_time=True
#     )
# )

# cot_explainer = SimpleExplainer(client, tokenizer=model.tokenizer,cot=True,logits=True,activations=False,max_tokens=500)
# explainer_out_dir = "saved_explanations/gpt2_cot"

# asyncio.run(
#     execute_model(
#         cot_explainer, 
#         explainer_inputs,
#         output_dir=explainer_out_dir,
#         record_time=True
#     )
# )
