import itertools

base_cmd = "python /mnt/ssd-1/alexm/sae-auto-interp/run_score_with_random_expls.py "

cfg_ranges = {
    "n_feats": [300],
    "feat_layer": [8, 32, 41],
    "n_train": [10],
    "n_test": [30],
    "n_explanations": [5],
    "kl_threshold": [0.1, 0.33, 1.0, 3.0],
    "random_explanations": [False],
    "latents": ["sae"],
    # "zero_ablate": [True],
}

keys, values = zip(*cfg_ranges.items())
cfgs = [dict(zip(keys, v)) for v in itertools.product(*values)]

for cfg in cfgs:
    cmd = base_cmd 
    run_name = "_".join([
        "".join([word[0] for word in k.split("_")]) + "=" + str(v)
        for k, v in cfg.items()
    ])
    run_path = f"/mnt/ssd-1/alexm/sae-auto-interp/counterfactual_results/{run_name}/generations.json"
    cmd += f" --expl_path \"{run_path}\" "
    cmd += " --explainer_name \"meta-llama/Meta-Llama-3.1-8B\" "
    print(cmd)
