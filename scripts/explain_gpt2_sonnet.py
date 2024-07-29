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

openrouter_key= "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"

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
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
        train_examples = random.sample(examples, 40)
        for example in train_examples:
            normalized_activations = (example.activations / record.max_activation)*10
            example.normalized_activations = normalized_activations.round()
        explainer_inputs.append(
            ExplainerInput(
                train_examples=train_examples,
                record=record
            )
        )

client = get_client("openrouter", "anthropic/claude-3.5-sonnet", api_key=openrouter_key)

simple_explainer = SimpleExplainer(client, tokenizer=model.tokenizer,cot=False,logits=False,activations=True)
explainer_out_dir = "saved_explanations/gpt2_claude"

asyncio.run(
    execute_model(
        simple_explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
        record_time=True
    )
)
