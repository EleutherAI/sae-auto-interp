from nnsight import LanguageModel 
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import CombinedStat, FeatureRecord, Logits, Activation
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]


model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="auto", dispatch=True)
submodule_dict = load_autoencoders(
    model, 
   layers,
    "saved_autoencoders/llama-exp32-v1",
)


tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features_llama_v1"
processed_features_path = "processed_features_llama_v1"

W_U = model.model.norm.weight * model.lm_head.weight

stats = CombinedStat(
    logits = Logits(
        model.tokenizer, 
        W_U = W_U
    ),
    activations = Activation(
        k = 1000,
    ),
)    
for layer, submodule in submodule_dict.items():
    ae = submodule.ae._module

    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        raw_dir=raw_features_path,
        selected_features=list(range(50)),
        max_examples=10000
    )
    
    for record in records:
        for example in record.examples:
            example.decode(model.tokenizer)

    stats.refresh(
        W_dec=ae.autoencoder.W_dec.T,
    )
    
    stats.compute(records, save_dir=processed_features_path)