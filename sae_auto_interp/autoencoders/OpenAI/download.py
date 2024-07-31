import blobfile as bf
import torch

location = "resid_post_mlp"

for layer_index in range(12):
    print(layer_index)

    path = f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}_v5_128k/autoencoders/{layer_index}.pt"

    with bf.BlobFile(path, mode="rb") as f:
        state_dict = torch.load(f)

        torch.save(
            state_dict,
            f"/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/OpenAI/gpt2_128k/{layer_index}.pt",
        )
