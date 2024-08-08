# %%
import torch
import torch.nn as nn
from nnsight import LanguageModel
from simple_parsing import ArgumentParser
import numpy as np
from functools import partial
from huggingface_hub import hf_hub_download

from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_filter, load_tokenized_data
from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents

class JumpReLUSAE(nn.Module):
  def __init__(self, d_model, d_sae):
    # Note that we initialise these to zeros because we're loading in pre-trained weights.
    # If you want to train your own SAEs then we recommend using blah
    super().__init__()
    self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
    self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
    self.threshold = nn.Parameter(torch.zeros(d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.b_dec = nn.Parameter(torch.zeros(d_model))

  def encode(self, input_acts):
    pre_acts = input_acts @ self.W_enc + self.b_enc
    mask = (pre_acts > self.threshold)
    acts = mask * torch.nn.functional.relu(pre_acts)
    return acts

  def decode(self, acts):
    return acts @ self.W_dec + self.b_dec

  def forward(self, acts):
    acts = self.encode(acts)
    recon = self.decode(acts)
    return recon


def main(cfg: CacheConfig):
    model = LanguageModel("google/gemma-2-2b", device_map="cuda:0", dispatch=True)

    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res",
        filename="layer_13/width_65k/canonical/params.npz",
        force_download=False,
    )

    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)

    def _forward(ae, x):
        return ae.encode(x)
    
    submodule = model.model.layers[13]
    submodule.ae = AutoencoderLatents(sae, partial(_forward, sae), 65_536)
    
    with model.edit(""):
        acts = submodule.output[0]
        submodule.ae(acts, hook=True)

    tokens = load_tokenized_data(
        cfg.ctx_len,
        model.tokenizer,
        "kh4dien/fineweb-100m-sample",
        "train[:15%]",
    )

    submodule_dict = {submodule._module_path: submodule}
    module_filter = {submodule._module_path: torch.arange(100)}

    cache = FeatureCache(
        model, submodule_dict, batch_size=cfg.batch_size, filters=module_filter
    )

    cache.run(cfg.n_tokens, tokens)

    cache.save_splits(
        n_splits=cfg.n_splits,
        save_dir="/share/u/caden/sae-auto-interp/weights/gemma",
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options

    main(cfg)
