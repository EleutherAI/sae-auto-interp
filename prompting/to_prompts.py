# %%

from raw_few_shot_examples import data
import random
import json

GENERATION_PROMPT = """Feature explanation: {explanation}

Text examples:

{examples}
"""

def unpack(which, n_test=3):

  samples = random.sample(sorted(data), n_test)

  prompts = []

  for feature in samples:
    s = data[feature]
    examples = [
      f"Example {i}: {s[which][i]['text']}"
      for i in range(len(s[which]))
    ]

    examples = "\n".join(examples)

    scores = {
      f"example_{i}": s[which][i]['score'] 
      for i in range(len(s[which]))
    }

    user_prompt = {
      "role": "user",
      "content": GENERATION_PROMPT.format(
        explanation=s["explanation"],
        examples=examples
      )
    }

    score_prompt = {
      "role": "assistant",
      "content": json.dumps(scores, indent=2)
    }

    prompts.extend([user_prompt, score_prompt])

  return prompts

all_examples = {}

for i in range(1,11):
    few_shot_examples = unpack("fuzzed" , i)
    all_examples[i] = few_shot_examples

with open("fuzzed_few_shot_examples.json", "w") as f:
    json.dump(all_examples, f, indent=2)
# %%
