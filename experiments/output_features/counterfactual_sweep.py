import itertools

base_cmd = "python /mnt/ssd-1/alexm/sae-auto-interp/counterfactual_pipeline.py "

cfg_ranges = {
    "n_feats": [300],
    "feat_layer": [8, 32, 41],
    "n_train": [10],
    "n_test": [30],
    "n_explanations": [5],
    "kl_threshold": [0.1, 0.33, 1.0, 3.0],
    "random_explanations": [False],
    "latents": ["sae", "random"],
    # "zero_ablate": [True],
}

keys, values = zip(*cfg_ranges.items())
cfgs = [dict(zip(keys, v)) for v in itertools.product(*values)]

for cfg in cfgs:
    cmd = base_cmd
    for k, v in cfg.items():
        if isinstance(v, bool):
            if v:
                cmd += f" --{k} "
        else:
            cmd += f" --{k} {v} "
    run_name = "_".join(
        [
            "".join([word[0] for word in k.split("_")]) + "=" + str(v)
            for k, v in cfg.items()
        ]
    )
    cmd += f' --run_prefix "{run_name}" '
    print(cmd)
