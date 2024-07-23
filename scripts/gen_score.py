from sae_auto_interp.scorers.generation.utils import score
from sae_auto_interp.autoencoders import load_oai_autoencoders
from nnsight import LanguageModel

model = LanguageModel("gpt2", device_map="cuda:0", dispatch=True)
submodule_dict = load_oai_autoencoders(
    model=model,
    ae_layers=[0, 2],
    weight_dir="sae_auto_interp/autoencoders/OpenAI/gpt2_128k"
)

score(
    model, 
    submodule_dict,
    "/share/u/caden/sae-auto-interp/results/generation",
)