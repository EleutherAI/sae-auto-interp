import asyncio
import random

from nnsight import LanguageModel

from sae_auto_interp.explainers import SimpleExplainer, ExplainerInput
from sae_auto_interp.clients import get_client, execute_model
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features import FeatureRecord
import argparse

argparser = argparse.ArgumentParser()


model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="cpu", dispatch=True)
tokens = load_tokenized_data(model.tokenizer,dataset_split="train",seq_len=256)

raw_features_path = "raw_features_llama_v2"
random.seed(22)
top_10_explainer_inputs = []
top_20_explainer_inputs = []
top_40_explainer_inputs = []
random_10_explainer_inputs = []
random_20_explainer_inputs = []
random_40_explainer_inputs = []
random_top_200_20_explainer_inputs = []
random_top_200_40_explainer_inputs = []
split_top_200_20_all_20_explainer_inputs = []


for layer in [24]:
    module_name =f".model.layers.{layer}"

    records = FeatureRecord.from_tensor(
        tokens,
        module_name,
        selected_features=list(range(0,500)),
        raw_dir = raw_features_path,
        min_examples=200,
        max_examples=10000
    )

    for record in records:

        examples = record.examples
        for example in examples:
            normalized_activations = (example.activations / record.max_activation)*10
            example.normalized_activations = normalized_activations.round()
        top_10 = examples[:10]
        top_20 = examples[:20]
        top_40 = examples[:40]
        top_200 = examples[:200]
        random_10 = random.sample(examples, 10)
        random_20 = random.sample(examples, 20)
        random_40 = random.sample(examples, 40)
        random_top_200_20 = random.sample(top_200, 20)
        random_top_200_40 = random.sample(top_200, 40)
        split_top_200_20_all_20 = random_top_200_20 + random_20

        top_10_explainer_inputs.append(ExplainerInput(
            train_examples=top_10,
            record=record
        ))
        top_20_explainer_inputs.append(ExplainerInput(
            train_examples=top_20,
            record=record
        ))
        top_40_explainer_inputs.append(ExplainerInput(
            train_examples=top_40,
            record=record
        ))
        
        random_10_explainer_inputs.append(ExplainerInput(
            train_examples=random_10,
            record=record
        ))
        random_20_explainer_inputs.append(ExplainerInput(
            train_examples=random_20,
            record=record
        ))
        random_40_explainer_inputs.append(ExplainerInput(
            train_examples=random_40,
            record=record
        ))
        random_top_200_20_explainer_inputs.append(ExplainerInput(
            train_examples=random_top_200_20,
            record=record
        ))
        random_top_200_40_explainer_inputs.append(ExplainerInput(
            train_examples=random_top_200_40,
            record=record
        ))
        split_top_200_20_all_20_explainer_inputs.append(ExplainerInput(
            train_examples=split_top_200_20_all_20,
            record=record
        ))
       



client = get_client("local", "casperhansen/llama-3-70b-instruct-awq", base_url="http://localhost:8001/v1")

explainer = SimpleExplainer(client, tokenizer=model.tokenizer,cot=False,logits=False,activations=True)
explainer_out_dir = "saved_explanations/llamav2_top_10"

# asyncio.run(
#     execute_model(
#         explainer, 
#         top_10_explainer_inputs,
#         output_dir=explainer_out_dir,
#         record_time=True
#     )
# )

# explainer_out_dir = "saved_explanations/llamav2_top_20"

# asyncio.run(
#     execute_model(
#         explainer, 
#         top_20_explainer_inputs,
#         output_dir=explainer_out_dir,
#         record_time=True
#     )
# )

# explainer_out_dir = "saved_explanations/llamav2_top_40"

# asyncio.run(
#     execute_model(
#         explainer, 
#         top_40_explainer_inputs,
#         output_dir=explainer_out_dir,
#         record_time=True
#     )
# )



explainer_out_dir = "saved_explanations/llamav2_random_10"

asyncio.run(
    execute_model(
        explainer, 
        random_10_explainer_inputs,
        output_dir=explainer_out_dir,
        record_time=True
    )
)

explainer_out_dir = "saved_explanations/llamav2_random_20"

asyncio.run(
    execute_model(
        explainer, 
        random_20_explainer_inputs,
        output_dir=explainer_out_dir,
        record_time=True
    )
)

explainer_out_dir = "saved_explanations/llamav2_random_40"

asyncio.run(
    execute_model(
        explainer, 
        random_40_explainer_inputs,
        output_dir=explainer_out_dir,
        record_time=True
    )
)

explainer_out_dir = "saved_explanations/llamav2_random_top_200_20"

asyncio.run(
    execute_model(
        explainer, 
        random_top_200_20_explainer_inputs,
        output_dir=explainer_out_dir,
        record_time=True
    )
)

explainer_out_dir = "saved_explanations/llamav2_random_top_200_40"

asyncio.run(
    execute_model(
        explainer, 
        random_top_200_40_explainer_inputs,
        output_dir=explainer_out_dir,
        record_time=True
    )
)

explainer_out_dir = "saved_explanations/llamav2_split_top_200_20_all_20"

asyncio.run(
    execute_model(
        explainer, 
        split_top_200_20_all_20_explainer_inputs,
        output_dir=explainer_out_dir,
        record_time=True
    )
)
