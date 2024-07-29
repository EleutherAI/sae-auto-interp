
from nnsight import LanguageModel
from sae_auto_interp.autoencoders import load_autoencoders
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data


model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
submodule_dict = load_autoencoders(
    model, 
    list(range(0,12,1)),
    "saved_autoencoders/gpt2_128k",
)

tokens = load_tokenized_data(model.tokenizer,dataset_split="train")

cache = FeatureCache(
    model, 
    submodule_dict,
    minibatch_size=64,
)
cache.run(tokens,n_tokens=10_000_000)

cache.save( save_dir="raw_features")
