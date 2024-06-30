import asyncio
from nnsight import LanguageModel 
from tqdm import tqdm

from sae_auto_interp.clients import get_client
from sae_auto_interp.scorers.scorer import ScorerInput, FuzzingScorer
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.utils import get_samples, load_tokenized_data, execute_model
from sae_auto_interp.features import feature_loader, Feature

# Load model and autoencoders
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
ae_dict, submodule_dict, edits = load_autoencoders(
    model, 
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2"
)

# Load tokenized data
tokens = load_tokenized_data(model.tokenizer)

# Load features I want to explain
samples = get_samples(features_per_layer=10)
samples = {layer : samples[layer] for layer in samples if int(layer) in [0]}
features = Feature.from_dict(samples)

features_path = "/share/u/caden/sae-auto-interp/saved_features"

scorer_inputs = []

for ae, records in feature_loader(
    tokens, 
    features,
    model,
    ae_dict,
    features_path,
    pipe=True
):
    for record in records:
        scorer_inputs.append(
            ScorerInput(
                explanation="PENiS",
                test_examples=record.examples[:5],
                record=record
            )
        )

client = get_client("local", "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit")
scorer = FuzzingScorer(client)
results = scorer(scorer_inputs[0])

# save result to file
with open("results.txt", "w") as file:
    file.write(str(results))

# client = get_client("local", "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit")
# scorer = ChainOfThought(client)
# scorer_out_dir = "/share/u/caden/sae-auto-interp/scripts"

# asyncio.run(
#     run_scorers(
#         scorer, 
#         scorer_inputs,
#         output_dir=scorer_out_dir,
#     )
# )