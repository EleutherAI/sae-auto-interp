import umap
import torch

class Backend:
    pass

class UmapBackend(Backend):
    embedding = None

    def refresh(self, W_dec=None, **kwargs):
        umap_model = umap.UMAP(
            n_neighbors=15, 
            metric='cosine', 
            min_dist=0.05, 
            n_components=2, 
            random_state=42
        )
        self.embedding = umap_model.fit_transform(W_dec)

class LogitBackend(Backend):
    logits = None

    def refresh(self, W_U=None, W_dec=None, **kwargs):
        self.logits = torch.matmul(W_U, W_dec).detach().cpu()
