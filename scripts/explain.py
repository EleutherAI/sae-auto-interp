import asyncio
from nnsight import LanguageModel 
from tqdm import tqdm

from sae_auto_interp.explainers import ChainOfThought, ExplainerInput
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import execute_model, load_tokenized_data, get_samples
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import CombinedStat, Logits, feature_loader, Feature

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

stats = CombinedStat(
    logits = Logits(
        model=model,
        top_k_logits=10
    )
)    

explainer_inputs = []

for ae, records in feature_loader(
    tokens, 
    features,
    model,
    ae_dict,
    features_path,
    pipe=True
):
    stats.refresh(W_dec=ae.decoder.weight)
    stats.compute(records)

    for record in records:
        explainer_inputs.append(
            ExplainerInput(
                record.examples[:10],
                record
            )
        )

client = get_client("local", "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit")
explainer = ChainOfThought(client)
explainer_out_dir = "/share/u/caden/sae-auto-interp/saved_explanations/caden"

asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
    )
)