# %%
import torch
locations_path = f"/share/u/caden/sae-auto-interp/raw_features/layer0_locations.pt"
activations_path = f"/share/u/caden/sae-auto-interp/raw_features/layer0_activations.pt" 

locations = torch.load(locations_path)
activations = torch.load(activations_path)
# %%
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.utils import load_tokenized_data
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

tokens = load_tokenized_data(
    model.tokenizer
)
records = FeatureRecord.from_tensor(
    tokens,
    model.tokenizer,
    0,
    "/share/u/caden/sae-auto-interp/raw_features/layer0_locations.pt",
    "/share/u/caden/sae-auto-interp/raw_features/layer0_activations.pt",
    max_examples=1000
)