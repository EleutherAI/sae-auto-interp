from nnsight import LanguageModel
import os
os.environ["CONFIG_PATH"] = "configs/pythia.yaml"
from sae_auto_interp.autoencoders import load_autoencoders
from sae_auto_interp.features import FeatureCache
from scripts_two.utils import load_module_stuff

model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map="auto", dispatch=True)

submodule_dict = load_autoencoders(
    model,
    None,
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/Sam/pythia-70m-deduped",
)


module_filter = load_module_stuff("feature-circuits/experiments/sfc.json")
cache = FeatureCache(model, submodule_dict, filters=module_filter)
cache.run()

cache.save( save_dir="/share/u/caden/sae-auto-interp/raw_features")




