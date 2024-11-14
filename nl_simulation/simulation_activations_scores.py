import asyncio
import json
import os
import time

import numpy as np
import orjson
import torch
from safetensors.numpy import load_file
from simple_parsing import ArgumentParser
from tqdm import tqdm

from sae_auto_interp.features.features import Example, Feature, FeatureRecord
import glob
import pandas as pd


    

def build_feature_record(selected_tokens ,activation,explanation=None,order=-1,sentence_idx=-1,feature=0):
    feature = Feature("layer", feature)
    activations = torch.zeros_like(selected_tokens,dtype=torch.float32) # The real activations don't matter
    activations[-1] = activation.round(decimals=3)
    example = Example(selected_tokens, activations)
    feature_record = FeatureRecord(feature)
    feature_record.explanation = explanation
    feature_record.test = [example]
    feature_record.sentence_idx = sentence_idx
    feature_record.order = order
    return feature_record

def create_records(sentence_idx, tokens, locations, activation, all_explanations,all_scores, window_size,SCORES_FOLDER):
    window_idx = np.random.randint(window_size+1,256-window_size)
    selected_window = tokens[sentence_idx][window_idx-window_size:window_idx]
    
    selected_sentence_idx = locations[:,0]==sentence_idx
    sentence_locations = locations[selected_sentence_idx]
    sentence_activation = activation[selected_sentence_idx]
    
    token_idx = sentence_locations[:,1]==window_idx-1 # selected tokens does not include the last token
    locations_at_token = sentence_locations[token_idx]
    activation_at_token = sentence_activation[token_idx]
    
    active_features = locations_at_token[:,2]

    # sort the active features by the activation
    sort_mask = torch.argsort(activation_at_token)
    sorted_active_features = active_features[sort_mask]

    feature_records = []
    # Create records for the active features
    for idx,active_feature in enumerate(sorted_active_features):
        if str(active_feature.item()) not in all_explanations:
            continue
        if str(active_feature.item()) in all_scores:
            score = "Score: "+str(all_scores[str(active_feature.item())])
        else:
            score = ""
        current_activation = activation_at_token[sort_mask[idx]]
        explanation = all_explanations[str(active_feature.item())]
        explanation = explanation+score
        feature_record = build_feature_record(selected_window,activation=current_activation,explanation=explanation,order=idx,sentence_idx=sentence_idx,feature=active_feature.item())
        feature_records.append(feature_record)
    
    os.makedirs(SCORES_FOLDER+f"{sentence_idx}", exist_ok=True)
    return feature_records


async def feature_generator(tokens, locations, activation, all_explanations,all_scores, window_size, number_of_sentences, start_sentence, SCORES_FOLDER):
    for sentence_idx in range(start_sentence,start_sentence+number_of_sentences):
        feature_records = create_records(sentence_idx, tokens, locations, activation, all_explanations,all_scores, window_size, SCORES_FOLDER)
        for feature_record in feature_records:
            yield feature_record
# Make postprocess for the fuzzing scorer
def scorer_postprocess(result,output_folder):
    predicted_quantile = result.score[0].predicted_quantile
    activation = result.score[0].activation.item()
    
    sentence_idx = result.record.sentence_idx
    order = result.record.order 
    feature = str(result.record.feature).split("feature")[-1]
    saving_result = {"feature":int(feature),"predicted_quantile":predicted_quantile,"activation":activation,"order":order}
    with open(f"{output_folder}{sentence_idx}/{feature}.txt", "wb") as f:
            f.write(orjson.dumps(saving_result))
    return result
    
def merge_results(SCORES_FOLDER):
    all_files = glob.glob(os.path.join(SCORES_FOLDER, "*.txt"))
    all_results = []
    for file in all_files:
        with open(file, "r") as f:
            data = json.load(f)
            all_results.append(data)
    
    all_data = pd.DataFrame(all_results)
    all_data.to_csv(f"{SCORES_FOLDER}/all_results.csv", index=False)
    # # remove the files
    # for file in all_files:
    #     os.remove(file)
    return all_data


def main(args):
    
    #model = LanguageModel("google/gemma-2-9b", device_map="cpu", dispatch=True,torch_dtype="float16")
    layer = 11
    all_locations = []
    all_activations = []
    ranges = ["0_26213","26214_52427","52428_78642","78643_104856","104857_131071"]
    for valid_range in ranges:
        split_data = load_file(f"/mnt/ssd-1/gpaulo/SAE-Zoology/raw_features/gemma/131k/.model.layers.{layer}/{valid_range}.safetensors")
        activations = torch.tensor(split_data["activations"])
        locations = torch.tensor(split_data["locations"].astype(np.int64))
        locations[:,2] = locations[:,2]+int(valid_range.split("_")[0])
        all_locations.append(locations)
        all_activations.append(activations)


    locations = torch.cat(all_locations)
    activation = torch.cat(all_activations)
    if args.explanation == "quantiles":
        with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/explanations_131k/model.layers.{layer}_feature.json", "r") as f:
            #load json file
            all_explanations = json.load(f)
    else:
        with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/explanations_131k_top/model.layers.{layer}_feature.json", "r") as f:
            #load json file
            all_explanations = json.load(f)
    all_scores={}
    if args.score == "fuzz":
        with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/formated_scores/gemma/131k/res/fuzz_layer_{layer}.json", "r") as f:
            #load json file
            all_scores = json.load(f)
    if args.score == "recall":
        with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/formated_scores/gemma/131k/res/recall_layer_{layer}.json", "r") as f:
            #load json file
            all_scores = json.load(f)

    
    if args.model_size == "8b":
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/activations/8b{args.score}{args.soft}/{args.window_size}/{args.explanation}/"
    elif args.model_size == "70b":
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/activations/70b{args.score}/{args.window_size}/{args.explanation}/"
    elif args.model_size == "8b-quantiles_top5":
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/activations/8b-quantiles_top5{args.score}/{args.window_size}/{args.explanation}/"
    elif args.model_size == "8b-top_top5":
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/activations/8b-top_top5{args.score}/{args.window_size}/{args.explanation}/"
    else:
        raise ValueError("Model size not supported")
    start = time.time()
    window_size = args.window_size
    np.random.seed(42)

        

    all_data = []
    for i in range(args.start_sentence,args.start_sentence+args.num_sentences):
        data = merge_results(f"{SCORES_FOLDER}{i}")
        all_data.append(data)
    all_data = pd.concat(all_data)
    # sort the locations by the locations[:,2]
    start = time.time()
    locations = locations.to("cuda")
    activation = activation.to("cuda")
    index_sort = torch.argsort(locations[:,2])
    target_locations = locations[index_sort]
    target_activation = activation[index_sort]
    all_data = all_data.reset_index(drop=False)
    # find all unique features
    unique_features = all_data["feature"].unique()
    for feature in tqdm(unique_features):
        start = time.time()
        #select the locations, knowing that locations is sorted
        # Since locations is sorted by feature number (locations[:,2]), 
        # we can use searchsorted to find the start and end indices
        start_idx = torch.searchsorted(target_locations[:,2], feature)
        end_idx = torch.searchsorted(target_locations[:,2], feature + 1)
        
        # Get all activations for this feature
        feature_activation = target_activation[start_idx:end_idx]
        feature_locations = target_locations[start_idx:end_idx]
        
        
        #print("Selecting features took: ",time.time()-start)
        # sort the activations
        start = time.time()
        sorted_activations = torch.sort(feature_activation)[0]
        #print("Sorting activations took: ",time.time()-start)
        linearly_spaced_activations = torch.linspace(sorted_activations[0],sorted_activations[-1],50)
        density_spaced_activations = sorted_activations[::len(sorted_activations)//50]
        #print("Creating linearly and density spaced activations took: ",time.time()-start)
        # for each activation in all_data, find the closest activation in the linearly and density spaced activations
        start = time.time()
        feature_data = all_data[all_data["feature"]==feature]
        for idx,row in feature_data.iterrows():
            #print(idx)
            closest_linear = (linearly_spaced_activations - row["activation"]).abs().argmin().item()
            closest_density = (density_spaced_activations - row["activation"]).abs().argmin().item()
            all_data.at[idx,"closest_linear"] = closest_linear
            all_data.at[idx,"closest_density"] = closest_density
            #break
        #print("Finding closest activations took: ",time.time()-start)
    all_data.to_csv(f"{SCORES_FOLDER}/all_results_with_quantiles.csv", index=False)

    end = time.time()
    # print(f"Time taken: {end-start}")

    
if __name__ == "__main__":
    parser = ArgumentParser()
    #parser.add_argument("--sentence_idx", type=int, default=0)
    parser.add_argument("--explanation",type=str,default="quantiles")
    parser.add_argument("--model_size", type=str, default="8b")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--num_sentences", type=int, default=100)
    parser.add_argument("--start_sentence", type=int, default=0)
    parser.add_argument("--score", type=str, default="")
    parser.add_argument("--soft", type=str, default="")
    args = parser.parse_args()
    #sentence_idx = args.sentence_idx

    main(args)