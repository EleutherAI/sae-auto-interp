import asyncio
import orjson


from functools import partial
from sae_auto_interp.explainers import SimpleExplainer
from sae_auto_interp.scorers import RecallScorer, FuzzingScorer
from sae_auto_interp.clients import Local
from sae_auto_interp.utils import (
    load_tokenized_data,
    load_tokenizer,
    default_constructor,
)
from sae_auto_interp.features import top_and_quantiles, FeatureLoader, FeatureDataset
from sae_auto_interp.pipeline import Pipe, Pipeline
from functools import partial
from sae_auto_interp.config import FeatureConfig

### Set directories ###

RAW_FEATURES_PATH = "raw_features/pythia"
EXPLAINER_OUT_DIR = "results/pythia_explanations"
recall_dir = "results/pythia_recall"
fuzz_dir = "results/pythia_fuzz"

def main(batch_size: int):
    ### Load dataset ###

    tokenizer = load_tokenizer("gpt2")
    tokens = load_tokenized_data(tokenizer)

    dataset = FeatureDataset(
        raw_dir=RAW_FEATURES_PATH,
        cfg=FeatureConfig(),
    )

    loader = FeatureLoader(
        tokens=tokens,
        dataset=dataset,
        constructor=partial(default_constructor, n_random=5, ctx_len=20, max_examples=5_000),
        sampler=top_and_quantiles,
    )

    ### Load client ###

    client = Local("meta-llama/Meta-Llama-3-8B-Instruct")

    ### Build Explainer pipe ###
    def explainer_postprocess(result):
        with open(f"{EXPLAINER_OUT_DIR}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.explanation))

    explainer_pipe = Pipe(
        SimpleExplainer(
            client, 
            tokenizer=tokenizer, 
            cot=False, 
            # activations=True,
            postprocess=explainer_postprocess,
        ),
        name="explainer",
    )

    ### Build Scorer pipe ###

    def scorer_preprocess(result):
        record = result.record
        
        record.explanation = result.explanation
        record.extra_examples = record.random_examples

        return record

    def scorer_postprocess(result, score_dir):
        with open(f"{score_dir}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    # scorer_pipe = Pipe(
    #     Actor(
    #         RecallScorer(client, tokenizer=tokenizer, batch_size=batch_size),
    #         preprocess=scorer_preprocess,
    #         postprocess=partial(scorer_postprocess, score_dir=recall_dir),
    #     ),
    #     Actor(
    #         FuzzingScorer(client, tokenizer=tokenizer, batch_size=batch_size),
    #         preprocess=scorer_preprocess,
    #         postprocess=partial(scorer_postprocess, score_dir=fuzz_dir),
    #     ),
    #     name="scorer",
    # )

    ### Build the pipeline ###

    pipeline = Pipeline(
        loader.load,
        explainer_pipe,
        # scorer_pipe,
    )

    asyncio.run(pipeline.run())


if __name__ == "__main__":

    main(5)
