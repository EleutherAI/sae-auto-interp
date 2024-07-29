import blobfile as bf
import torch

location = "resid_post_mlp"
for layer_index in range(0,12):
    with bf.BlobFile(f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}_v5_32k/autoencoders/{layer_index}.pt", mode="rb") as f:
        state_dict = torch.load(f)
        torch.save(state_dict, f"saved_autoencoders/gpt2_32k/{layer_index}.pt")