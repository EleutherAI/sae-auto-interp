
from nnsight import LanguageModel
from transformers import AutoTokenizer
from sae_auto_interp.autoencoders import load_autoencoders
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data
import argparse

model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="auto")
argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]

submodule_dict,model = load_autoencoders(
    model, 
    layers,
    "saved_autoencoders/llama-exp32-v1",
)

tokens = load_tokenized_data(model.tokenizer,dataset_split="train",seq_len=256)
print(submodule_dict)
cache = FeatureCache(
    model, 
    submodule_dict,
    minibatch_size=16,
)
cache.run(tokens,n_tokens=10_000_000)

cache.save( save_dir="raw_features_llama_v1")
