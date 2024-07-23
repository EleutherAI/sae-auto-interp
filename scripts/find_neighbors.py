from nnsight import LanguageModel
import torch
from sae_auto_interp.autoencoders import load_oai_autoencoders
from sae_auto_interp.scorers.neighbor.utils import get_neighbors

NEIGHBOR_DIR = "sae_auto_interp/scorers/neighbor"

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
submodule_dict = load_oai_autoencoders(
    model, 
    [0,2],
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/OpenAI/gpt2_128k",
)

modules = [".transformer.h.0", ".transformer.h.2"]
features = {
    m : list(range(20)) for m in modules
}

get_neighbors(
    submodule_dict,
    features,
    NEIGHBOR_DIR,
)