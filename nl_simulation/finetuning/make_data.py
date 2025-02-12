import torch
import numpy as np
import json
from sae_auto_interp.utils import load_tokenized_data
from safetensors.numpy import load_file
from nnsight import LanguageModel
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset
from simple_parsing import ArgumentParser

PROMPT = "<|start_header_id|>user<|end_header_id|>\n[EXPLANATION]: {explanation}\n[SENTENCE]: {sentence}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n"

class CustomDataset(Dataset):
    def __init__(self, layer,tokens, source_tokenizer, target_tokenizer, start_index, top_k=None,explanation=None):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.tokens = tokens
        self.start_index = start_index
        if top_k == -1:
            self.top_k = None
        else:
            self.top_k = top_k
        self.load_locations(layer)
        if explanation == "quantiles":
            with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/explanations_131k/model.layers.{layer}_feature.json", "r") as f:
                #load json file
                all_explanations = json.load(f)
        else:
            with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/explanations_131k_top/model.layers.{layer}_feature.json", "r") as f:
                #load json file
                all_explanations = json.load(f)
        self.explanations = all_explanations


    def load_locations(self, layer):
        all_locations = []
        all_activations = []
        ranges = ["0_26213","26214_52427","52428_78642","78643_104856","104857_131071"]
        for valid_range in ranges:
            split_data = load_file(f"/mnt/ssd-1/gpaulo/SAE-Zoology/raw_features/gemma/131k/.model.layers.{layer}/{valid_range}.safetensors")
            locations = torch.tensor(split_data["locations"].astype(np.int64))
            activations = torch.tensor(split_data["activations"].astype(np.float32))
            all_locations.append(locations)
            all_activations.append(activations)
        
        self.locations = torch.cat(all_locations)
        self.activations = torch.cat(all_activations)
        self.length = locations[0,:].max().item()
        self.start_index = start_index
    def get_next(self,sentence_idx,window_idx,window_size):
        selected_tokens = self.tokens[sentence_idx][window_idx-window_size:window_idx]
        interesting_idx = self.locations[:,0]==sentence_idx
        interesting_locations = self.locations[interesting_idx]
        token_idx = interesting_locations[:,1]==window_idx-1 # because the last token is not included in the window
        locations_at_token = interesting_locations[token_idx]
        activations_at_token = self.activations[interesting_idx][token_idx]
        active_features = locations_at_token[:,2] # I get all active to get a number 
        if self.top_k is not None:
            top_k_activations = activations_at_token.topk(self.top_k).indices
            selected_features = locations_at_token[top_k_activations,2] # I select only the top k
        else:
            selected_features = active_features
        
        sentence = self.source_tokenizer.decode(selected_tokens)
        
        tokenized = []
        labels = []
        for active_feature in selected_features:
            if str(active_feature.item()) not in self.explanations:
                continue
            
            explanation = self.explanations[str(active_feature.item())]
            
            templated = PROMPT.format(explanation=explanation, sentence=sentence)
            tokenized.append(templated)
            labels.append(1)

        # Like this i get a ratio of 30/5 or similar, which is still lower than real data but not bad
        random_features = np.random.choice(np.arange(131072), size=len(active_features), replace=False)

        for non_active_feature in random_features:
            if non_active_feature in active_features:
                continue
            if str(non_active_feature) not in self.explanations:
                continue
            explanation = self.explanations[str(non_active_feature)]
            templated = PROMPT.format(explanation=explanation, sentence=sentence)
            
            tokenized.append(templated)
            labels.append(0)
        
        return tokenized, labels


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # This method should return a single processed sample
        # Select 5 random windows positions 
        window_size = np.random.choice(range(16,32))
        window_idx = np.random.choice(np.arange(window_size,256), size=5, replace=False)
        all_samples = []
        all_labels = []
        for window_idx in window_idx:
                sample, labels = self.get_next(idx+self.start_index,window_idx,window_size)
                all_samples.extend(sample)
                all_labels.extend(labels)
        return all_samples, all_labels

args = ArgumentParser()
args.add_argument("--explanation", type=str, default="quantiles")
args.add_argument("--top_k", type=int, default=5)
args.add_argument("--layer_train", type=int, default=12)
args.add_argument("--layer_test", type=int, default=14)
args = args.parse_args()

model = LanguageModel("google/gemma-2-9b", device_map="cpu", dispatch=True,torch_dtype="float16")
new_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

tokens = load_tokenized_data(
        256,
        model.tokenizer,
        "EleutherAI/rpj-v2-sample",
        "train[:1%]",
        "",    
        "raw_content"
    )
start_index = 1000

train_dataset = CustomDataset(layer=args.layer_train,tokens=tokens, source_tokenizer=model.tokenizer, target_tokenizer=new_tokenizer, start_index=start_index,top_k=args.top_k,explanation=args.explanation)
test_dataset = CustomDataset(layer=args.layer_test,tokens=tokens, source_tokenizer=model.tokenizer, target_tokenizer=new_tokenizer, start_index=start_index+len(train_dataset),explanation=args.explanation)
all_samples_train = []
all_labels_train = []
for i in tqdm(range(500)):

    samples_from_sentence, labels_from_sentence = train_dataset[i]
    all_samples_train.extend(samples_from_sentence)
    all_labels_train.extend(labels_from_sentence)
all_samples_test = []
all_labels_test = []
for i in tqdm(range(200)):
    samples_from_sentence, labels_from_sentence = test_dataset[i]
    all_samples_test.extend(samples_from_sentence)
    all_labels_test.extend(labels_from_sentence)

dataset = Dataset.from_dict({"samples": all_samples_train, "labels": all_labels_train})
dataset_test = Dataset.from_dict({"samples": all_samples_test, "labels": all_labels_test})

dataset = dataset.train_test_split(test_size=0.1)
dataset_test = dataset_test.train_test_split(test_size=0.1)
dataset["test"] = dataset_test["train"]
name = f"simulation_data_{args.explanation}_top{args.top_k}"
dataset.save_to_disk(name)
