
from nnsight import LanguageModel
from transformers import AutoTokenizer
from sae_auto_interp.autoencoders import load_autoencoders
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data
model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="cuda", dispatch=True)
_,submodule_dict = load_autoencoders(
    model, 
    [24],
    "saved_autoencoders/llama-exp32-v2",
)

tokens = load_tokenized_data(model.tokenizer,dataset_split="train",seq_len=512)
cache = FeatureCache(
    model, 
    submodule_dict,
    minibatch_size=8,
)
cache.run(tokens,n_tokens=10_000_000)

cache.save( save_dir="raw_features_llama_512")
