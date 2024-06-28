DSCORER_SYSTEM_PROMPT = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain feature of text, such as "male pronouns" or "text with negative sentiment".

You will then be given several text examples. For each example in turn, assign it a score according to the following rules:
0 – The feature is completely irrelevant throughout the context (relative to the base distribution of the internet).
1 – The feature is related to the context, but not near the highlighted text or only vaguely related.
2 – The feature is only loosely related to the highlighted text or related to the context near the highlighted text.
3 – The feature cleanly identifies the activating text.

Return your answer in JSON format. 
"""

# https://www.neuronpedia.org/gpt2-small/10-res-jb/12307
DSCORER_EXAMPLE_ONE = """Feature explanation: The word "mark" in the context of significant events

Text examples:

Example 1: broadcast.ĊĊWhile the episode will mark C.K.'s debut as an
Example 2:brate and honour the LGBTQ community" to denote the start of Pride Month.ĊĊ
Example 3: Darkfall players.ĊĊThis will mark the start of our active promotion of the
Example 4:," he said in a statement."Today marks a victory for American freedom over Presidential tyranny
Example 5:in's speech at the Republican National Convention was her first major address as the Republican vice
"""


DSCORER_RESPONSE_ONE = """{
  "example_1":3,
  "example_2":2,
  "example_3":3,
  "example_4":3,
  "example_5":2
}"""

# https://www.neuronpedia.org/gpt2-small/9-res-jb/7677
# https://www.neuronpedia.org/gpt2-small/9-res-jb/1
DSCORER_EXAMPLE_TWO = """Feature explanation: years stated in a copyright notice

Text examples:

Example 1: this long to complete a report for an incident like this is inexcusable.âĢ
Example 2:Could the right tool be built? In the late 2000s, some people at Mozilla
Example 3: The Washington Times, LLC. Click here
Example 4: this article and others.ĊĊÂ© 2017 WTOP. All Rights Reserved.<|endoftext|>
Example 5: around roads, and as Malaysians, you know lah we got our own per
"""


DSCORER_RESPONSE_TWO = """{
  "example_1":0,
  "example_2":0,
  "example_3":2,
  "example_4":3,
  "example_5":0
}"""

# https://www.neuronpedia.org/gpt2-small/10-res-jb/8743
# https://www.neuronpedia.org/gpt2-small/10-res-jb/4259
DSCORER_EXAMPLE_THREE = """Feature explanation: Organizations related to government, policy, and advocacy.

Text examples:

Example 1: a senior fellow at the liberal Center for American Progress, is more concerned by TuesdayâĢ
Example 2:ell MP, The Secretary of State for Scotland (Dumfriesshire,
Example 3: in both chambers agree on establishing nonprofit health care cooperatives and stripping insurance companies of an
Example 4: at the London-based International Institute for Environmental and Development (IIED).ĊĊ
Example 5: walk awayĊĊMen have none of those rights. The Jezebel article spells
"""


DSCORER_RESPONSE_THREE = """{
  "example_1":3,
  "example_2":2,
  "example_3":1,
  "example_4":3,
  "example_5":1
}"""

USER_PROMPT = """Feature explanation: {explanation}

Text examples:

{examples}
"""


def get_detection_template(examples, explanation):
  user_prompt = USER_PROMPT.format(explanation=explanation, examples=examples)

  prompt = [
    {"role": "system", "content": DSCORER_SYSTEM_PROMPT},
    {"role": "user", "content": DSCORER_EXAMPLE_ONE},
    {"role": "assistant", "content": DSCORER_RESPONSE_ONE},
    {"role": "user", "content": DSCORER_EXAMPLE_TWO},
    {"role": "assistant", "content": DSCORER_RESPONSE_TWO},
    {"role": "user", "content": DSCORER_EXAMPLE_THREE},
    {"role": "assistant", "content": DSCORER_RESPONSE_THREE},
    {"role": "user", "content": user_prompt}
  ]

  return prompt