import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


# This is from the GemmaScope tutorial
# https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp#scrollTo=WYfvS97fAFzq
class JumpReluSae(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        # print(pre_acts.shape)
        # print(torch.nonzero(mask).shape)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

    @classmethod
    def from_pretrained(cls, path: str, type: str, device: str) -> nn.Module:
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-9b-pt-" + type,
            filename=f"{path}/params.npz",
            force_download=False,
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
        model = cls(params["W_enc"].shape[0], params["W_enc"].shape[1])
        model.load_state_dict(pt_params)
        if device == "cuda":
            model.cuda()
        return model
