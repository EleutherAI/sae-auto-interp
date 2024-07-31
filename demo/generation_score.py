from nnsight import LanguageModel

from sae_auto_interp.autoencoders import load_oai_autoencoders
from sae_auto_interp.scorers.generation.utils import score


def main():
    model = LanguageModel("gpt2", device_map="auto", dispatch=True)

    submodule_dict = load_oai_autoencoders(
        model=model, ae_layers=list(range(0, 12, 2)), weight_dir="weights/gpt2_128k"
    )

    score(
        model,
        submodule_dict,
        "/share/u/caden/sae-auto-interp/results/generation",
    )


if __name__ == "__main__":
    main()
