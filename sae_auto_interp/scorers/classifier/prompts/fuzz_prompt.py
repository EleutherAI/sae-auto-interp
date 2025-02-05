# %%
DSCORER_SYSTEM_PROMPT = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain feature of text, such as "male pronouns" or "text with negative sentiment". You will be given a few examples of text that contain this feature. Portions of the sentence which strongly represent this feature are between tokens << and >>.

Some examples might be mislabeled. Your task is to determine if every single token within << and >> is correctly labeled. Consider that all provided examples could be correct, none of the examples could be correct, or a mix. An example is only correct if every marked token is representative of the feature

For each example in turn, return 1 if the sentence is correctly labeled or 0 if the tokens are mislabeled. You must return your response in a valid Python list. Do not return anything else besides a Python list.
"""

# https://www.neuronpedia.org/gpt2-small/6-res-jb/6048
DSCORER_EXAMPLE_ONE = """Feature explanation: Words related to American football positions, specifically the tight end position.

Text examples:

Example 0:<|endoftext|>Getty ImagesĊĊPatriots<< tight end>> Rob Gronkowski had his bossâĢĻ
Example 1: posted<|endoftext|>You should know this<< about>> offensive line coaches: they are large, demanding<< men>>
Example 2: Media Day 2015ĊĊLSU<< defensive>> end Isaiah Washington (94) speaks<< to the>>
Example 3:<< running backs>>," he said. .. Defensive<< end>> Carroll Phillips is improving and his injury is
Example 4:<< line>>, with the left side âĢĶ namely<< tackle>> Byron Bell at<< tackle>> and<< guard>> Amini
"""

# DSCORER_RESPONSE_ONE = """{
#   "example_0": 1,
#   "example_1": 0,
#   "example_2": 0,
#   "example_3": 1,
#   "example_4": 1
# }"""

DSCORER_RESPONSE_ONE = "[1,0,0,1,1]"

# https://www.neuronpedia.org/gpt2-small/6-res-jb/9396
DSCORER_EXAMPLE_TWO = """Feature explanation: The word "guys" in the phrase "you guys".

Text examples:

Example 0: if you are<< comfortable>> with it. You<< guys>> support me in many other ways already and
Example 1: birth control access<|endoftext|> but I assure you<< women>> in Kentucky aren't laughing as they struggle
Example 2:âĢĻs gig! I hope you guys<< LOVE>> her, and<< please>> be nice,
Example 3:American, told<< Hannity>> that âĢľyou<< guys>> are playing the race card.âĢĿ
Example 4:<< the>><|endoftext|>ľI want to<< remind>> you all that 10 days ago (director Massimil
"""

# DSCORER_RESPONSE_TWO = """{
#   "example_0": 0,
#   "example_1": 0,
#   "example_2": 0,
#   "example_3": 0,
#   "example_4": 0
# }"""

DSCORER_RESPONSE_TWO = "[0,0,0,0,0]"

# https://www.neuronpedia.org/gpt2-small/8-res-jb/12654
DSCORER_EXAMPLE_THREE = """Feature explanation: "of" before words that start with a capital letter.

Text examples:

Example 0: climate, TomblinâĢĻs Chief<< of>> Staff Charlie Lorensen said.Ċ
Example 1: no wonderworking relics, no true Body and Blood<< of>> Christ, no true Baptism
Example 2:ĊĊDeborah Sathe, Head<< of>> Talent Development and Production at Film London,
Example 3:ĊĊIt has been devised by Director<< of>> Public Prosecutions (DPP)
Example 4: and fair investigation not even include the Director<< of>> Athletics? Â· Finally, we believe the
"""

# DSCORER_RESPONSE_THREE = """{
#   "example_0": 1,
#   "example_1": 1,
#   "example_2": 1,
#   "example_3": 1,
#   "example_4": 1
# }"""

DSCORER_RESPONSE_THREE = "[1,1,1,1,1]"

GENERATION_PROMPT = """Feature explanation: {explanation}

Text examples:

{examples}
"""


def prompt(examples, explanation):
    generation_prompt = GENERATION_PROMPT.format(
        explanation=explanation, examples=examples
    )

    defaults = [
        {"role": "user", "content": DSCORER_EXAMPLE_ONE},
        {"role": "assistant", "content": DSCORER_RESPONSE_ONE},
        {"role": "user", "content": DSCORER_EXAMPLE_TWO},
        {"role": "assistant", "content": DSCORER_RESPONSE_TWO},
        {"role": "user", "content": DSCORER_EXAMPLE_THREE},
        {"role": "assistant", "content": DSCORER_RESPONSE_THREE},
    ]

    prompt = [
        {"role": "system", "content": DSCORER_SYSTEM_PROMPT},
        *defaults,
        {"role": "user", "content": generation_prompt},
    ]

    return prompt
