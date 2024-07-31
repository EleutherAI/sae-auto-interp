GSCORER_SYSTEM_PROMPT = """You will be presented with a description of a certain feature of text.

Your task is to come up with {n_examples} examples of short text samples that contain the exact feature that was described to you.

Guidelines:
- You must pay extremely careful attention to the exact details of the feature being decribed.
- Every single example you write must possess the feature.
- Try to keep the examples varied.
- Unless the feature description explicitly refers to position within a sentence, you should make sure to vary the position in the sentence at which the feature appears. E.g. the word/token at which the feature appears should not always be at the start of the sentence.
- Return your response as a JSON object with keys "Example 0", "Example 1", ..., "Example N" and the corresponding examples as values.
- IMPORTANT: If the feature explanation involves some broader context of the text, you must establish the context at the start of each example. By the time the feature appears in the example, the context must already have been established.
"""

GSCORER_EXAMPLE_ONE = (
    """Description of text feature: male names in text to do with sports."""
)

GSCORER_RESPONSE_ONE = """{
    "example_0":"The olympic gold medal went to Bob.",
    "example_1":"Arsenal won the league that year. Gary Lineker was angry about it.",
    "example_2":"The next tennis match will be between Andy Murray and Roger Federer.",
    "example_3":"The greatest NBA player of all time was Michael Jordan, no debate.",
    "example_4":"The Warriors beat the Nuggets 104-102 in overtime, with a clutch 3-pointer from Stephen Curry yet again.",
    "example_5":"When you think of hockey, you think of Wayne Gretzky.",
    "example_6":"WHY DO LIVERPOOL KEEP LOSING. WHAT HAPPENED TO SCORING GOALS?? FUCK JURGEN KLOPP. FUCK LIVERPOOL",
    "example_7":"Yet another superbowl for the best QB in the NFL: Patrick Mahomes.",
    "example_8":"He's boxing like Mike Tyson in his prime.",
    "example_9":"Top scorer of all time: Lionel Messi."
}"""


GSCORER_EXAMPLE_TWO = """Description of text feature: phrases starting with 'only to'"""

GSCORER_RESPONSE_TWO = """{
    "example_0":"to make a Broadway flop, only to see it become a smashing success. What",
    "example_1":"on the west side of Phoenix, only to have election officials close the site around 7",
    "example_2":"peak or a backcountry landmark, only to prompt a response like âĢľYeah,",
    "example_3":"stranded and starving in the Arctic, only to encounter a mythological beast who pops up",
    "example_4":"zel makes the deal gladly, only to find herself enchanted and stuck in a tower.",
    "example_5":"if his side gig was lucrative, only to realize he was selling the figures like hot",
    "example_6":"to rocket their way to victory, only to be met with derision by fans for",
    "example_7":"teaching position at a Beijing college âĢĵ only to have the entire department cut a year later",
    "example_8":"voted for more dam construction, only to be stymied by Byzantine and politicized",
    "example_9":"a woman who flew to Albuquerque, only to be blocked by protesters. By the time"
}"""


def get_gen_scorer_template(explanation, n_examples):
    prompt = [
        {
            "role": "system",
            "content": GSCORER_SYSTEM_PROMPT.format(n_examples=n_examples),
        },
        {"role": "user", "content": GSCORER_EXAMPLE_ONE},
        {"role": "assistant", "content": GSCORER_RESPONSE_ONE},
        {"role": "user", "content": GSCORER_EXAMPLE_TWO},
        {"role": "assistant", "content": GSCORER_RESPONSE_TWO},
        {"role": "user", "content": f"Description of text feature: {explanation}"},
    ]

    return prompt
