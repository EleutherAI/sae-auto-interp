from dataclasses import dataclass


@dataclass
class ExplainerInterventionExample:
    prompt: str
    top_tokens: list[str]
    top_p_increases: list[float]

    def __post_init__(self):
        self.prompt = self.prompt.replace("\n", "\\n")

    def text(self) -> str:
        tokens_str = ", ".join(f"'{tok}' (+{round(p, 3)})" for tok, p in zip(self.top_tokens, self.top_p_increases))
        return f"<PROMPT>{self.prompt}</PROMPT>\nMost increased tokens: {tokens_str}"
    
@dataclass
class ExplainerNeuronFormatter:
    intervention_examples: list[ExplainerInterventionExample]
    explanation: str | None = None

    def text(self) -> str:
        text = "\n\n".join(example.text() for example in self.intervention_examples)
        text += "\n\nExplanation:"
        if self.explanation is not None:
            text += " " + self.explanation
        return text


def get_explainer_prompt(neuron_prompter: ExplainerNeuronFormatter, few_shot_examples: list[ExplainerNeuronFormatter] | None = None) -> str:
    prompt = "We're studying neurons in a transformer model. We want to know how intervening on them affects the model's output.\n\n" \
        "For each neuron, we'll show you a few prompts where we intervened on that neuron at the final token position, and the tokens whose logits increased the most.\n\n" \
        "The tokens are shown in descending order of their probability increase, given in parentheses. Your job is to give a short summary of what outputs the neuron promotes.\n\n"
    
    i = 1
    for few_shot_example in few_shot_examples or []:
        assert few_shot_example.explanation is not None
        prompt += f"Neuron {i}\n" + few_shot_example.text() + "\n\n"
        i += 1

    prompt += f"Neuron {i}\n"
    prompt += neuron_prompter.text()

    return prompt


fs_examples = [
    ExplainerNeuronFormatter(
        intervention_examples=[
            ExplainerInterventionExample(
                prompt="Given 4x is less than 10,",
                top_tokens=[" 4", " 10", " 40", " 2"],
                top_p_increases=[0.11, 0.04, 0.02, 0.01]
            ),
            ExplainerInterventionExample(
                prompt="For some reason",
                top_tokens=[" one", " 1", " fr"],
                top_p_increases=[0.14, 0.01, 0.01]
            ),
            ExplainerInterventionExample(
                prompt="insurance does not cover claims for accounts with",
                top_tokens=[" one", " more", " 10"],
                top_p_increases=[0.10, 0.02, 0.01]
            )
        ],
        explanation="numbers"
    ),
    ExplainerNeuronFormatter(
        intervention_examples=[
            ExplainerInterventionExample(
                prompt="Once",
                top_tokens=[" upon", " in", " a", " long"],
                top_p_increases=[0.22, 0.2, 0.05, 0.04]
            ),
            ExplainerInterventionExample(
                prompt="Ryan Quarles\\n\\nRyan Francis Quarles (born October 20, 1983)",
                top_tokens=[" once", " happily", " for"],
                top_p_increases=[0.03, 0.31, 0.01]
            ),
            ExplainerInterventionExample(
                prompt="MSI Going Full Throttle @ CeBIT",
                top_tokens=[" Once", " once", " in", " the", " a", " The"],
                top_p_increases=[0.02, 0.01, 0.01, 0.01, 0.01, 0.01]
            ),
        ],
        explanation="storytelling"
    ),
    ExplainerNeuronFormatter(
        intervention_examples=[
            ExplainerInterventionExample(
                prompt="My favorite food is",
                top_tokens=[" oranges", " bananas", " apples"],
                top_p_increases=[0.81, 0.09, 0.029]
            ),
            ExplainerInterventionExample(
                prompt=" to eat",
                top_tokens=[" fro", " fruit", " oranges", " bananas", " strawberries"],
                top_p_increases=[0.14, 0.13, 0.11, 0.1, 0.032]
            ),
            ExplainerInterventionExample(
                prompt="",
                top_tokens=["Oranges", " the", " oranges", " sweet", "Apple", "\"", " apples", " a", " bananas", " red"],
                top_p_increases=[0.038, 0.025, 0.024, 0.019, 0.018, 0.017, 0.016, 0.015, 0.014, 0.013],
            ),
            ExplainerInterventionExample(
                prompt="Whenever they would see",
                top_tokens=[" fruit", " a", " apples", " red"],
                top_p_increases=[0.09, 0.06, 0.06, 0.05]
            ),
        ],
        explanation="fruits and vegetables"
    ),
]

scorer_separator = "<PASSAGE>\n"


def get_scorer_surprisal_prompt(prompt, generation, explanation, few_shot_prompts=None, few_shot_explanations=None, few_shot_generations=None, return_explanation_start=False) -> str:
    if few_shot_explanations is not None:
        assert few_shot_generations is not None and few_shot_prompts is not None
        assert len(few_shot_explanations) == len(few_shot_generations) == len(few_shot_prompts)
        few_shot_prompt = "\n\n".join(get_scorer_surprisal_prompt(txt, gen, expl) for txt, expl, gen in zip(few_shot_prompts, few_shot_explanations, few_shot_generations)) + "\n\n"
    else:
        few_shot_prompt = ""

    expl_text = f"{explanation}\""
    text = few_shot_prompt + f"{scorer_separator}{prompt}{generation}\n\nThe above passage contains an amplified amount of \"" + expl_text
    expl_start_idx = len(text) - len(expl_text)
    if return_explanation_start:
        return text, expl_start_idx  # type: ignore
    return text


few_shot_prompts = [
    "from west to east, the westmost of the seven",
    "Given 4x is less than 10,",
    "In information theory, the information content, self-information, surprisal, or Shannon information is a basic quantity derived",
    "My favorite food is",
]
few_shot_explanations = [
    "Asia",
    "numbers",
    "she/her pronouns",
    "fruits and vegetables",
]
few_shot_generations = [
    " wonders of the world is the great wall of china",
    " 4",
    " by her when she was a student at Windsor",
    " oranges",
]