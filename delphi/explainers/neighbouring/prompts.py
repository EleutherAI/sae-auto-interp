SYSTEM = """You are a meticulous AI researcher conducting an important investigation into patterns found in language.
Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text correct examples on which words are selected with delimiters like <<this>>.
An entire sequence of tokens can be contained between delimiters <<just like this>>. 
These words are important to define the pattern you are trying to identify.
How important each word is for the behavior is listed after each example in parentheses.
You will also be given a list of extra examples, that don't have the special words, but are similar to the correct examples.
These can be used to provide more context about the pattern, and help you make the explanation more specific.

- Try to produce a concise final description. 
- Don't focus only on the special words, but also on the context in which they appear and that surround them.
- If the examples are uninformative, you don't need to mention them. 
- Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Do not make lists of possible explanations. Keep your explanations short and concise.
- The last line of your response must be the formatted explanation, using [EXPLANATION]:

{prompt}
"""


### EXAMPLE 1 ###

EXAMPLE_1 = """
Correct example 1:  and he was <<over the moon>> to find
Correct example 2:  we'll be laughing <<till the cows come home>>! Pro
Correct example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd

Extra example 1:  and he was playing in the field.
Extra example 2:  the cows were grazing in the green pastures.
Extra example 3:  I was looking for something but my eyes just couldn't see it.
"""

EXAMPLE_1_ACTIVATIONS = """
Correct example 1:  and he was <<over the moon>> to find
Activations: ("over", 5), (" the", 6), (" moon", 9)
Correct example 2:  we'll be laughing <<till the cows come home>>! Pro
Activations: ("till", 5), (" the", 5), (" cows", 8), (" come", 8), (" home", 8)
Correct example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd
Activations: ("than", 5), (" meets", 7), (" the", 6), (" eye", 8)

Extra example 1:  and he was playing in the field.
Extra example 2:  the cows were grazing in the green pastures.
Extra example 3:  I was looking for something but my eyes just couldn't see it.
"""

EXAMPLE_1_EXPLANATION = """
[EXPLANATION]: Common idioms in text conveying positive sentiment.
"""

### EXAMPLE 2 ###

EXAMPLE_2 = """
Correct example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Correct example 2:  every year you get tall<<er>>," she
Correct example 3:  the hole was small<<er>> but deep<<er>> than the

Extra example 1:  the server brought our dinner to the wrong table
Extra example 2:  My sister works as a teacher in elementary school
Extra example 3:  carpenter built a beautiful wooden cabinet.
"""

EXAMPLE_2_ACTIVATIONS = """
Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Activations: ("er", 8)
Example 2:  every year you get tall<<er>>," she
Activations: ("er", 2)
Example 3:  the hole was small<<er>> but deep<<er>> than the
Activations: ("er", 9), ("er", 9)

Extra example 1:  the server brought our dinner to the wrong table
Extra example 2:  My sister works as a teacher in elementary school
Extra example 3:  carpenter built a beautiful wooden cabinet.
"""

EXAMPLE_2_EXPLANATION = """
[EXPLANATION]: The token "er" at the end of a comparative adjective describing size.
"""

### EXAMPLE 3 ###

EXAMPLE_3 = """
Example 1:  something happening inside my <<house>>", he
Example 2:  presumably was always contained in <<a box>>", according
Example 3:  people were coming into the <<smoking area>>".

However he
Example 4:  Patrick: "why are you getting in the << way?>>" Later,
"""

EXAMPLE_3_ACTIVATIONS = """
Example 1:  something happening inside my <<house>>", he
Activations: ("house", 7)
Example 2:  presumably was always contained in <<a box>>", according
Activations: ("a", 5), ("box", 9)
Example 3:  people were coming into the <<smoking area>>".

However he
Activations: ("smoking", 2), ("area", 4)
Example 4:  Patrick: "why are you getting in the << way?>>" Later,
Activations: ("way", 4), ("?", 2)
"""


EXAMPLE_3_EXPLANATION = """
[EXPLANATION]: Nouns representing a distinct objects that contains something, sometimes preciding a quotation mark.
"""


def get(item):
    return globals()[item]


def _prompt(n, activations=False, **kwargs):
    starter = (
        get(f"EXAMPLE_{n}") if not activations else get(f"EXAMPLE_{n}_ACTIVATIONS")
    )

    prompt_atoms = [starter]

    return "".join(prompt_atoms)


def _response(n, cot=False, **kwargs):
    response_atoms = []
    if cot:
        response_atoms.append(get(f"EXAMPLE_{n}_COT_ACTIVATION_RESPONSE"))

    response_atoms.append(get(f"EXAMPLE_{n}_EXPLANATION"))

    return "".join(response_atoms)


def example(n, **kwargs):
    prompt = _prompt(n, **kwargs)
    response = _response(n, **kwargs)

    return prompt, response


def system(cot=False):
    prompt = ""

    if cot:
        prompt += COT

    return [
        {
            "role": "system",
            "content": SYSTEM.format(prompt=prompt),
        }
    ]


def system_single_token():
    return [{"role": "system", "content": SYSTEM_SINGLE_TOKEN}]
