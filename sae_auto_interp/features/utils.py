from torch import Tensor
import torch
from typing import List


def select_features(features:Tensor, indices:Tensor, feature_index:Tensor) -> tuple[Tensor,Tensor]:
    return features[indices[:,2] == feature_index], indices[indices[:,2] == feature_index]


def get_activated_sentences(features:Tensor, indices:Tensor, feature_index:Tensor, all_tokens:List[Tensor],max_selection:int=50) -> tuple[Tensor,Tensor,Tensor]:
    selected_features, selected_indices = select_features(features, indices, feature_index)
    if selected_features.shape[0] < max_selection:
        max_selection = selected_features.shape[0]
    else:
        max_selection = max_selection
    top_activations = selected_features.topk(max_selection)
    
    #These are the top activations, but we want to also get other activations in the same sentences
    top_indices = selected_indices[top_activations.indices]
    
    unique_top_sentence_idx = top_indices[:,0].unique()
    all_activations = torch.tensor([])
    all_indices = torch.tensor([])
    #Get all activations in the same sentences
    for counter,idx in enumerate(unique_top_sentence_idx):
        wanted_indexes = torch.nonzero(selected_indices[:,0] == idx)
        activations = selected_features[wanted_indexes]
        all_activations = torch.cat((all_activations,activations))
        new_indices = torch.clone(selected_indices[wanted_indexes])
        new_indices[:,:,0] = counter
        all_indices = torch.cat((all_indices,new_indices))
    all_activations = all_activations.squeeze(1)
    all_indices = all_indices.squeeze(1)
    sentences = all_tokens[unique_top_sentence_idx.int()]
    return sentences, all_activations, all_indices
