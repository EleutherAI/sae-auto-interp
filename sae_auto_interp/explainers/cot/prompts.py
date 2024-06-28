SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into a certain neuron in a language model. Your task is to analyze the neuron and provide a explanation that thoroughly encapsulates its behavior. Your task comes in two parts:

(Part 1) Tokens that the neuron activates highly on in text

You will be given a list of text examples on which the neuron activates. The specific tokens which cause the neuron to activate will appear between delimiters like {l}this{r}. If a sequence of consecutive tokens all cause the neuron to activate, the entire sequence of tokens will be contained between delimiters {l}just like this{r}.

Step 1: The activating tokens, tokens before those tokens, and following tokens will be listed. Look for patterns in these tokens.
Step 2: Write down several general shared features of the text examples.

(Part 2) Tokens that the neuron boosts in the next token prediction 

You will also be shown a list called Top_logits. The logits promoted by the neuron shed light on how the neuron's activation influences the model's predictions or outputs. Look at this list of Top_logits and refine your hypotheses from part 1. It is possible that this list is more informative than the examples from part 1. 

Step 3: Pay close attention to the words in this list and write down what they have in common.  
Step 4: Look at what they have in common, as well as patterns in the tokens you found in Part 1, to produce a single explanation for what features of text cause the neuron to activate. Propose your explanation in the following format: 
[EXPLANATION]: <your explanation>

Guidelines: 
- Try to produce a concise final description. Simply describe the text features that activate the neuron, and what its role might be based on the tokens it predicts. 
- If either the text features or the predicted tokens are completely uninformative, you don't need to mention them.
- The last line of your response must be the formatted explanation.

Here are several example explanations."""


EXAMPLE_1 = """
Example 1:  and he was {l}over the moon{r} to find
Example 2:  we'll be laughing {l}till the cows come home{r}! Pro
Example 3:  thought Scotland was boring, but really there's more {l}than meets the eye{r}! I'd

Top_logits: ["elated", "joyful", "story", "thrilled", "spider"]

ACTIVATING TOKENS: "over the moon", "till the cows come home", "than meets the eye".
PREVIOUS TOKENS: "was", "laughing", "more".
NEXT TOKENS: "to", "!", "I'd", "!"."""


RESPONSE_1 = """
(Part 1)
Step 1.
The activating tokens are all parts of common idioms.
The previous tokens have nothing in common.
The next tokens are sometimes exclamation marks.

Step 2.
- The examples contain common idioms.
- In some examples, the activating tokens are followed by an exclamation mark.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities? Yes, I missed one:

- The text examples all convey positive sentiment.

(Part 2)
Step 3.
SIMILAR TOKENS: "elated", "joyful", "thrilled".
The top logits list contains words that are strongly associated with positive emotions.

Step 4.
[EXPLANATION]: Common idioms in text conveying positive sentiment."""


EXAMPLE_2 = """
Example 1:  a river is wide but the ocean is wider{l}. {r}The ocean
Example 2:  every year you get tall{l}er{r}," she
Example 3:  the hole was small{l}er{r} but deep{l}er{r} than the

Top_logits: ["apple", "running", "book", "wider", "quickly"]

ACTIVATING TOKENS: ". ", "er", "er", "er".
PREVIOUS TOKENS: "er", "tall", "small", "deep".
NEXT TOKENS: "The", ",", " but", "than"."""


RESPONSE_2 = """
(Part 1)
Step 1.
The activating tokens are mostly "er".
The previous tokens are mostly adjectives, or parts of adjectives, describing size.
The next tokens have nothing in common.
The neuron seems to activate on, or near, the token "er" in comparative adjectives describing size.

Step 2.
- In each example, the activating token was "er" appearing at the end of a comparative adjective.
- The comparative adjectives ("wider", "taller", "smaller", "deeper") all describe size.

Let me look again for patterns in the examples. Are there any links or hidden linguistic commonalities that I missed? I can't see any.

(Part 2)
Step 3.
SIMILAR TOKENS: None.
The top logits list contains mostly unrelated nouns and adverbs.

Step 4.
[EXPLANATION]: The token "er" at the end of a comparative adjective describing size."""


EXAMPLE_3 = """
Example 1:  something happening inside my {l}house{r}", he
Example 2:  presumably was always contained in {l}a box{r}", according
Example 3:  people were coming into the {l}smoking area{r}".

However he
Example 4:  Patrick: "why are you getting in the {l} way?{l}" Later,

Top_logits: ["room", "end", "container, "space", "plane"]

ACTIVATING TOKENS: "house", "a box", "smoking area", " way?".
PREVIOUS TOKENS: "my", "in", "the", "the".
NEXT TOKENS: all quotation marks."""


RESPONSE_3 = """
(Part 1)
Step 1.
The activating tokens are all things that one can be in.
The previous tokens have nothing in common.
The next tokens are all quotation marks.

Step 2.
- The examples involve being inside something, sometimes figuratively.
- The activating token is a thing which something else is inside of.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities? Yes, I missed one:

- The activating token is followed by a quotation mark, suggesting it occurs within speech.

(Part 2)
Step 3.
SIMILAR TOKENS: "room", "container", "space".
The top logits list suggests a focus on nouns representing physical or metaphorical spaces.

Step 4.
[EXPLANATION]: Nouns preceding a quotation mark, representing a thing that contains something."""

USER_START = """
{examples}
Top_logits: {top_logits}

ACTIVATING TOKENS: {activating}.
PREVIOUS TOKENS: {previous}.
NEXT TOKENS: {following}."""

opening_prompt = f"{SYSTEM_PROMPT}\n{EXAMPLE_1}\n{RESPONSE_1}\n{EXAMPLE_2}\n{RESPONSE_2}\n{EXAMPLE_3}\n{RESPONSE_3}\n{USER_START}"

def create_prompt(l, r, examples, top_logits, activating, previous, following, simplifiy=False):
    user_prompt = opening_prompt.format(
        l=l,
        r=r,
        examples=examples,
        top_logits=top_logits,
        activating=activating,
        previous=previous,
        following=following
    )

    simplified_user_prompt = USER_START.format(
        examples=examples,
        top_logits=top_logits,
        activating=activating,
        previous=previous,
        following=following
    )

    if simplifiy:
        return simplified_user_prompt, user_prompt
    return user_prompt