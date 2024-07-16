DSCORER_SYSTEM_PROMPT = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain feature of text, such as "male pronouns" or "text with negative sentiment".

You will then be given several text examples. Your task is to determine which examples possess the feature.

For each example in turn, return 1 if the sentence is correctly labeled or 0 if the tokens are mislabeled. You must return your response in a json format. For example, if you believe the first and last examples are correct, but the middle three are not, you would return :"{\n  \"example_0\": 1,\n  \"example_1\": 0,\n  \"example_2\": 0,\n  \"example_3\": 0,\n  \"example_4\": 1\n}.
"""

# https://www.neuronpedia.org/gpt2-small/6-res-jb/6048
DSCORER_EXAMPLE_ONE = """Feature explanation: Words related to American football positions, specifically the tight end position.

Text examples:

Example 0:<|endoftext|>Getty ImagesĊĊPatriots tight end Rob Gronkowski had his bossâĢĻ
Example 1: names of months used in The Lord of the Rings:ĊĊâĢľâĢ¦the
Example 2: Media Day 2015ĊĊLSU defensive end Isaiah Washington (94) speaks to the
Example 3: shown, is generally not eligible for ads. For example, videos about recent tragedies,
Example 4: line, with the left side âĢĶ namely tackle Byron Bell at tackle and guard Amini
"""

DSCORER_RESPONSE_ONE = """{
  "example_0": 1,
  "example_1": 0,
  "example_2": 0,
  "example_3": 1,
  "example_4": 1
}"""

# https://www.neuronpedia.org/gpt2-small/6-res-jb/9396
DSCORER_EXAMPLE_TWO = """Feature explanation: The word "guys" in the phrase "you guys".

Text examples:

Example 0: enact an individual health insurance mandate?âĢĿ, Pelosi's response was to dismiss both
Example 1: birth control access<|endoftext|> but I assure you women in Kentucky aren't laughing as they struggle
Example 2: du Soleil Fall Protection Program with construction requirements that do not apply to theater settings because
Example 3:Ċ<|endoftext|> distasteful. Amidst the slime lurk bits of Schadenfre
Example 4: the<|endoftext|>ľI want to remind you all that 10 days ago (director Massimil
"""

DSCORER_RESPONSE_TWO = """{
  "example_0": 0,
  "example_1": 0,
  "example_2": 0,
  "example_3": 0,
  "example_4": 0
}"""

# https://www.neuronpedia.org/gpt2-small/8-res-jb/12654
DSCORER_EXAMPLE_THREE = """Feature explanation: "of" before words that start with a capital letter.

Text examples:

Example 0: climate, TomblinâĢĻs Chief of Staff Charlie Lorensen said.Ċ
Example 1: no wonderworking relics, no true Body and Blood of Christ, no true Baptism
Example 2:ĊĊDeborah Sathe, Head of Talent Development and Production at Film London,
Example 3:ĊĊIt has been devised by Director of Public Prosecutions (DPP)
Example 4: and fair investigation not even include the Director of Athletics? Â· Finally, we believe the
"""

DSCORER_RESPONSE_THREE = """{
  "example_0": 1,
  "example_1": 1,
  "example_2": 1,
  "example_3": 1,
  "example_4": 1
}"""

GENERATION_PROMPT = """Feature explanation: {explanation}

Text examples:

{examples}
"""


from .clean_few_shot_examples import examples as clean_examples

def prompt(examples, explanation, n_test=-1):
  generation_prompt = GENERATION_PROMPT.format(explanation=explanation, examples=examples)

  defaults = [
    {"role": "user", "content": DSCORER_EXAMPLE_ONE},
    {"role": "assistant", "content": DSCORER_RESPONSE_ONE},
    {"role": "user", "content": DSCORER_EXAMPLE_TWO},
    {"role": "assistant", "content": DSCORER_RESPONSE_TWO},
    {"role": "user", "content": DSCORER_EXAMPLE_THREE},
    {"role": "assistant", "content": DSCORER_RESPONSE_THREE},
  ]

  if n_test == 0:
    defaults = []
  elif n_test != -1:
    defaults = clean_examples[str(n_test)]

  prompt = [
    {"role": "system", "content": DSCORER_SYSTEM_PROMPT},
    *defaults,
    {"role": "user", "content": generation_prompt}
  ]

  return prompt