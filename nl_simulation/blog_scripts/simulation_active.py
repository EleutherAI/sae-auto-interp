import asyncio
import json
import os
import time
from functools import partial

import numpy as np
import orjson
import torch
from nnsight import LanguageModel
from safetensors.numpy import load_file
from simple_parsing import ArgumentParser
from nl.fuzz import FuzzingScorer

from sae_auto_interp.clients import Offline
from sae_auto_interp.features.features import Example, Feature, FeatureRecord
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.utils import load_tokenized_data
import glob
import pandas as pd

def build_feature_record(selected_tokens ,active=True,explanation=None,order=-1,sentence_idx=-1,feature=0):
    feature = Feature("layer", feature)
    activations = torch.zeros_like(selected_tokens) # The real activations don't matter
    example = Example(selected_tokens, activations)
    feature_record = FeatureRecord(feature)
    feature_record.explanation = explanation
    if active:
        feature_record.test = [example]
    else:
        feature_record.extra_examples = [example]
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
    sorted_active_features = active_features[torch.argsort(activation_at_token)]

    feature_records = []
    # Create records for the active features
    for idx,active_feature in enumerate(sorted_active_features):
        if str(active_feature.item()) not in all_explanations:
            continue
        if str(active_feature.item()) in all_scores:
            score = "Score: "+str(all_scores[str(active_feature.item())])
        else:
            score = ""
        explanation = all_explanations[str(active_feature.item())]
        explanation = explanation+score
        feature_record = build_feature_record(selected_window,explanation=explanation,active=True,order=idx,sentence_idx=sentence_idx,feature=active_feature.item())
        feature_records.append(feature_record)
    # Create records for the non-active features
    random_features = np.random.choice(np.arange(131072), size=1000, replace=False)
    for non_active_feature in random_features:
        if non_active_feature in active_features:
            continue
        if str(non_active_feature) not in all_explanations:
            continue
        explanation = all_explanations[str(non_active_feature)]
        if str(non_active_feature) in all_scores:
            score = "Score: "+str(all_scores[str(non_active_feature)])
        else:
            score = ""
        explanation = explanation+score
        feature_record = build_feature_record(selected_window,explanation=explanation,active=False,order=-1,sentence_idx=sentence_idx,feature=non_active_feature)
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
    prediction = result.score[0].prediction
    ground_truth = result.score[0].ground_truth
    conditional_probability = result.score[0].conditional_probability
    probability = result.score[0].probability
    #explanation = result.record.explanation
    #text = result.score[0].text
    sentence_idx = result.record.sentence_idx
    order = result.record.order 
    feature = str(result.record.feature).split("feature")[-1]
    saving_result = {"feature":int(feature),"prediction":prediction,"ground_truth":ground_truth,"conditional_probability":conditional_probability,"probability":probability,"order":order}
    with open(f"{output_folder}{sentence_idx}/{feature}.txt", "wb") as f:
            f.write(orjson.dumps(saving_result))
    return result
    
def coalest_results(SCORES_FOLDER):
    all_files = glob.glob(os.path.join(SCORES_FOLDER, "*.txt"))
    all_results = []
    for file in all_files:
        with open(file, "r") as f:
            data = json.load(f)
            all_results.append(data)
    
    all_data = pd.DataFrame(all_results)
    all_data.to_csv(f"{SCORES_FOLDER}/all_results.csv", index=False)
    # remove the files
    for file in all_files:
        os.remove(file)


def main(args):
    
    model = LanguageModel("google/gemma-2-9b", device_map="cpu", dispatch=True,torch_dtype="float16")
    layer = 11
    all_locations = []
    all_activations = []
    if args.sae_size == "131k":
        ranges = ["0_26213","26214_52427","52428_78642","78643_104856","104857_131071"]
        for valid_range in ranges:
            split_data = load_file(f"/mnt/ssd-1/gpaulo/SAE-Zoology/raw_features/gemma/131k/.model.layers.{layer}/{valid_range}.safetensors")
            activations = torch.tensor(split_data["activations"])
            locations = torch.tensor(split_data["locations"].astype(np.int64))
            locations[:,2] = locations[:,2]+int(valid_range.split("_")[0])
            all_locations.append(locations)
            all_activations.append(activations)
    if args.sae_size == "16k":
        ranges = ["0_3275","3276_6552","6553_9829","9830_13106","13107_16383"]
        for valid_range in ranges:
            split_data = load_file(f"/mnt/ssd-1/gpaulo/SAE-Zoology/raw_features/gemma/16k/.model.layers.{layer}/{valid_range}.safetensors")
            activations = torch.tensor(split_data["activations"])
            locations = torch.tensor(split_data["locations"].astype(np.int64))
            locations[:,2] = locations[:,2]+int(valid_range.split("_")[0])
            all_locations.append(locations)
            all_activations.append(activations)
    

    locations = torch.cat(all_locations)
    activation = torch.cat(all_activations)
    if args.sae_size == "131k":
        if args.explanation == "quantiles":
            with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/explanations/model.layers.{layer}_feature.json", "r") as f:
                #load json file
                all_explanations = json.load(f)
        else:
            with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/explanations_131k_top/model.layers.{layer}_feature.json", "r") as f:
                #load json file
                all_explanations = json.load(f)
    else:
        if args.explanation == "quantiles":
            with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/explanations_16k/model.layers.{layer}_feature.json", "r") as f:
                #load json file
                all_explanations = json.load(f)
        else:
            raise ValueError("Explanation top not supported")
    all_scores={}
    if args.score == "fuzz":
        with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/formated_scores/gemma/{args.sae_size}/res/fuzz_layer_{layer}.json", "r") as f:
            #load json file
            all_scores = json.load(f)
    if args.score == "recall":
        with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/formated_scores/gemma/{args.sae_size}/res/recall_layer_{layer}.json", "r") as f:
            #load json file
            all_scores = json.load(f)

    tokens = load_tokenized_data(
            256,
            model.tokenizer,
            "EleutherAI/rpj-v2-sample",
            "train[:1%]",
            "",    
            "raw_content"
        )

    if args.model_size == "8b":
        client = Offline("meta-llama/Meta-Llama-3.1-8B-Instruct",max_memory=0.8,max_model_len=5120,num_gpus=2,batch_size=4000)
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/scores/{args.sae_size}/8b{args.score}/{args.window_size}/{args.explanation}/"
    elif args.model_size == "70b":
        client = Offline("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",max_memory=0.75, max_model_len=4096,num_gpus=4)
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/scores/{args.sae_size}/70b{args.score}/{args.window_size}/{args.explanation}/"
    elif args.model_size == "8b-quantiles_top5":
        client = Offline("quantiles_top5",max_memory=0.8,max_model_len=5120,num_gpus=2,batch_size=4000)
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/scores/{args.sae_size}/8b-quantiles_top5{args.score}/{args.window_size}/{args.explanation}/"
    elif args.model_size == "8b-top_top5":
        client = Offline("top_top5",max_memory=0.8,max_model_len=5120,num_gpus=2,batch_size=4000)
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/scores/{args.sae_size}/8b-top_top5{args.score}/{args.window_size}/{args.explanation}/"
    else:
        raise ValueError("Model size not supported")
    start = time.time()
    window_size = args.window_size
    np.random.seed(42)

        
    if args.model_size == "8b-quantiles_top5" or args.model_size == "8b-lora-top":
        finetuned = True
    else:
        finetuned = False
    if args.score != "":
        score = True
    else:
        score = False
   

    fuzz_scorer = FuzzingScorer(client, model.tokenizer,contexts=False, finetuned=finetuned,score=score)
    # make folder for the scores
            
    scorer_pipe = Pipe(process_wrapper(fuzz_scorer, postprocess=partial(scorer_postprocess,output_folder=SCORES_FOLDER)))
    pipeline = Pipeline(
        feature_generator(tokens, locations, activation, all_explanations,all_scores, window_size, args.num_sentences, args.start_sentence, SCORES_FOLDER),
        scorer_pipe,
    )

    asyncio.run(pipeline.run(10000))

    for i in range(args.start_sentence,args.start_sentence+args.num_sentences):
        coalest_results(f"{SCORES_FOLDER}{i}")

    end = time.time()
    print(f"Time taken: {end-start}")

    
if __name__ == "__main__":
    parser = ArgumentParser()
    #parser.add_argument("--sentence_idx", type=int, default=0)
    parser.add_argument("--explanation",type=str,default="quantiles")
    parser.add_argument("--model_size", type=str, default="8b")
    parser.add_argument("--sae_size",type=str,default="131k")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--num_sentences", type=int, default=100)
    parser.add_argument("--start_sentence", type=int, default=0)
    parser.add_argument("--score", type=str, default="")
    args = parser.parse_args()
    #sentence_idx = args.sentence_idx

    main(args)