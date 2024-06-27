from torch import Tensor
import torch
from typing import List


def select_features(features:Tensor, indices:Tensor, feature_index:Tensor) -> tuple[Tensor,Tensor]:
    return features[indices[:,2] == feature_index], indices[indices[:,2] == feature_index]

def cluster_features(selected_features,selected_indices):
    unique_sentences = torch.unique(selected_indices[:,0])
    sentence_features = []
    sum = []
    indices = []
    for i,sentence in enumerate(unique_sentences):
        sentence_indices = torch.nonzero(selected_indices[:,0] == sentence)
        selected = selected_features[sentence_indices]
        sentence_features.append(selected.squeeze(1))
        sum.append(selected.sum())
        indices.append(selected_indices[sentence_indices].squeeze(1))
    return sentence_features,sum,indices

def get_activated_sentences(features:Tensor, indices:Tensor, feature_index:Tensor, all_tokens:List[Tensor],max_selection:int=50) -> tuple[Tensor,Tensor,Tensor]:
    selected_features, selected_indices = select_features(features, indices, feature_index)
    selected_features = selected_features.cuda()
    selected_indices = selected_indices.cuda()
    sentence_features, sum,sentence_indices = cluster_features(selected_features,selected_indices)
    
    if len(sentence_features) < max_selection:
        max_selection = len(sentence_features)
    else:
        max_selection = max_selection
    if max_selection == -1:
        top_activations = sentence_features
        top_activation_indices = range(len(sentence_features))
        max_selection = len(sentence_features)
    else:
        sum = torch.tensor(sum)
        top_activations = sum.topk(max_selection)
        top_activation_indices = top_activations.indices
        
    #These are the top activations, but we want to also get other activations in the same sentences
    
    all_activations = torch.tensor([]).cuda()
    all_indices = torch.tensor([]).cuda()
    sentences = torch.tensor([],dtype=torch.int64).cuda()
    #Get all activations in the same sentences
    for counter,idx in enumerate(top_activation_indices):
        t_features = sentence_features[idx]
        t_indices = sentence_indices[idx].clone()
        all_activations = torch.cat((all_activations,t_features))
        sentence = all_tokens[t_indices[0,0].int()].cuda()
        sentences = torch.cat((sentences,sentence))
        t_indices[:,0] = counter
        all_indices = torch.cat((all_indices,t_indices))
    sentences = sentences.reshape(max_selection,-1)

    return sentences.cpu(), all_activations.cpu(), all_indices.cpu()

