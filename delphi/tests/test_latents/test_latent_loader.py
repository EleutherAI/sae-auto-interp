
import torch

from delphi.latents.loader import LatentDataset


def test_latent_loader(latent_dataset: LatentDataset):
    """
    Test that the latent loader works correctly. Doesn't test constructor or sampler.
    """
    dataset = latent_dataset
    assert len(dataset) == 5
    assert dataset.buffers[0].latents == torch.tensor([0])
    assert dataset.buffers[1].latents == torch.tensor([7000])
    assert dataset.buffers[2].latents == torch.tensor([14000])
    assert dataset.buffers[3].latents == torch.tensor([21000])
    assert dataset.buffers[4].latents == torch.tensor([28000])
