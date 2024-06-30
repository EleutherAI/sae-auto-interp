DSCORER_SYSTEM_PROMPT = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain feature of text, such as "male pronouns" or "text with negative sentiment".

You will then be given several text examples. Activating examples are formatted between tokens like <<this>>.

The answer must be returned in JSON format. Mark correct exampels with 1 and incorrect examples with 0.
"""

DSCORER_EXAMPLE_ONE = """Feature explanation: The word "of" immediately before a capitalised word.

Text examples:

Example 1: climate, TomblinâĢĻs Chief of Staff Charlie Lorensen said.Ċ
Example 2: UMK, could be considered by Head of Delegation Guy Freeman and the BBC
Example 3:.ĊĊIn a statement, Director of National Intelligence James R. Clapper Jr.,
Example 4:, the CEO of giant asset manager BlackRock Inc, in an interview on Wednesday.
Example 5:andyburnhammp is running for Leader of the Labour Party https://t.co
"""


DSCORER_RESPONSE_ONE = """{
  "example_1": 1,
  "example_2": 1,
  "example_3": 1,
  "example_4": 0,
  "example_5": 0
}"""

DSCORER_EXAMPLE_TWO = """Feature explanation: male pronouns and names.

Text examples:

Example 1: "unfortunately", but we knew that later, a great deal of money would
Example 2: of the president, but after the process had stopped, she stated that
Example 3: Reuters (2017) 27th March 2018

---- CORRESPONDENT Jane Elizabeth
Example 4: every day for hours and hours, Sarah tried her hardest to
Example 5: (FOI) requests made in a bid to find out exactly who has the power
"""


DSCORER_RESPONSE_TWO = """{
  "example_1": 0,
  "example_2": 0,
  "example_3": 0,
  "example_4": 0,
  "example_5": 0
}"""

DSCORER_EXAMPLE_THREE = """Feature explanation: The word “care” in the context of health care policies and reform.

Text examples:

Example 1:. Does President Obama deserve credit for health care and other accomplishments?ĊĊA.
Example 2:delete any reference to 'C/- [Care of] Lippo".ĊĊ
Example 3: in both chambers agree on establishing nonprofit health care cooperatives and stripping insurance companies of an
Example 4: will delay a vote on their proposed health care legislation on June 27 at the Capitol.
Example 5: arguing for spending $36 million on health care for undocumented Oregon children during a Monday session
"""


DSCORER_RESPONSE_THREE = """{
  "example_1": 1,
  "example_2": 0,
  "example_3": 1,
  "example_4": 1,
  "example_5": 1
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