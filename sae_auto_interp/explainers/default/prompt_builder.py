from typing import List

from .prompts import example, system


def build_examples(
    **kwargs,
):
    examples = []

    for i in range(1, 4):
        prompt, response = example(i, **kwargs)

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "system",
                "content": response,
            },
        ]

        examples.extend(messages)

    return examples


def build_prompt(
    examples,
    activations: bool = False,
):
    
    messages = system(
        
    )

    few_shot_examples = build_examples(
    
        activations=activations,
    )

    messages.extend(few_shot_examples)

    user_start = f"\n{examples}\n"

    messages.append(
        {
            "role": "user",
            "content": user_start,
        }
    )

    return messages
