# %%
EXPLANATION_SYSTEM_PROMPT = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain explanation of a feature of text, such as "male pronouns" or "text with negative sentiment" and examples of text that may or may not contain this feature. Some explanations will be given a score from 0 to 1. The higher the score the better the explanation is, and you should be more certain of your response (positive or negative).

These features of text are normally identified by looking for specific words or patterns in the text. There are many features associated with a single token, but for any given token, less than 0.1 percent of the time is the feature related to the token.

Your job is to identify if the last token of the example, which is marked between << and >>, is significantly related to the feature. It is possible for a sentence to be related to the feature, but the last token not meaningfully related to the feature.

Return 1 only if the marked token is meaningfully related to the feature. Return 0 otherwise. You must return your response in a valid Python list. Do not return anything else besides a Python list.
"""



# https://www.neuronpedia.org/gpt2-small/6-res-jb/6048
EXPLANATION_EXAMPLE_ONE_A = """Feature explanation: Words related to American football positions, specifically the tight end position. {score}

Text examples:

Getty Images Patriots tight end Rob Gronkowski had his <<boss>>
"""

EXPLANATION_RESPONSE_ONE_A = "[0]"

EXPLANATION_EXAMPLE_ONE_B = """Feature explanation: Words related to American football positions, specifically the tight end position. {score}

Text examples:

line, with the left side namely tackle Byron Bell at tackle and<< guard>>
"""

EXPLANATION_RESPONSE_ONE_B = "[1]"


# https://www.neuronpedia.org/gpt2-small/6-res-jb/9396
EXPLANATION_EXAMPLE_TWO_A = """Feature explanation: The word "guys" in the phrase "you guys". {score}

Text examples:

if you are comfortable>> with it. You guys support me in many other ways already<< and>>
"""


EXPLANATION_RESPONSE_TWO_A = "[0]"

EXPLANATION_EXAMPLE_TWO_B = """Feature explanation: The word "guys" in the phrase "you guys". {score}

Text examples:

American, told Hannity that you guys are playing the race card<<.>>
"""


EXPLANATION_RESPONSE_TWO_B = "[0]"



# https://www.neuronpedia.org/gpt2-small/8-res-jb/12654
EXPLANATION_EXAMPLE_THREE_A = """Feature explanation: "of" before words that start with a capital letter. {score}

Text examples:

climate, Tomblin Chief of Staff Charlie Lorensen said<<.>>
"""


EXPLANATION_RESPONSE_THREE_A = "[0]"

EXPLANATION_EXAMPLE_THREE_B = """Feature explanation: "of" before words that start with a capital letter. {score}

Text examples:

no wonderworking relics, no true Body and Blood of <<Christ>>
"""


EXPLANATION_RESPONSE_THREE_B = "[1]"


GENERATION_EXPLANATION_PROMPT = """Feature explanation: {explanation}. {score}

Text examples:

{examples}
"""
# %%
EXPLANATION_SYSTEM_PROMPT_SIM = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain explanation of a feature of text, such as "male pronouns" or "text with negative sentiment" and examples of text that contains this feature. Some explanations will be given a score from 0 to 1. The higher the score the better the explanation is, and you should be more certain of your response (positive or negative).

These features of text are normally identified by looking for specific words or patterns in the text. There are many features associated with a single token, and sometimes the feature is related with the previous token.

Your job is to identify how much the the last token, which is marked between << and >>, represents the feature. You will output a integer between 0 and 9, where 0 is a no relation to the explanation and 9 is a strong relation.

Most of the tokens should have no relation. The ones that are related, should more likely be given 1 than 2, 2 than 3, and so on. Only give a 9 if the description exactly matches the token.

You must return your response in a valid Python list. Do not return anything else besides a Python list.
"""




# model.layers.10_feature3254 gemma 131k
EXPLANATION_EXAMPLE_ONE_A_SIM = """The term culture and its variations, often used in contexts that describe social, artistic, or national characteristics, customs, or traditions. {score}

Text examples:

Example 1: s central and expat friendly neighborhood at Spanish Panama. Spanish language immersion programs include airport pickup, tours and ecotourism, <<cultural>>
Example 2: This issue, like free trade, divides both parties along class lines. There is a strong opening for a <<culturally>> 
"""

EXPLANATION_RESPONSE_ONE_A_SIM = "[9,4]"


# 4834
EXPLANATION_EXAMPLE_TWO_A_SIM = """Adjectives with negative connotations describing situations, emotions, or personal relationships, often expressing a sense of conflict, tension, or strong emotions. {score}

Text examples:

Example 1: Panda wrote:B) McDonald has been anti metro from day one and this stinks of <<sour>>
"""

EXPLANATION_RESPONSE_TWO_A_SIM = "[1]"




# 6517
EXPLANATION_EXAMPLE_THREE_A_SIM = """Adjectives describing states of mind or perception that are captivating, mesmerizing, or induce a trance-like state, as well as nouns referring to narcotics, often used in contexts that suggest something is attention-grabbing or thought-provoking {score}

Text examples:
, a plate that dissolve into the image of the full moon, with an almost <<hypnotic>>
selfless self breaking back into the conventional world. It is only when this samad<<hi>>
"""

EXPLANATION_RESPONSE_THREE_A_SIM = "[6,1]"


GENERATION_EXPLANATION_PROMPT_SIM = """Feature explanation: {explanation}. {score}

Text examples:

{examples}
"""

# %%
CONTEXT_SYSTEM_PROMPT = """You are an intelligent and meticulous linguistics researcher.

You will be shown a set of texts that are related to a certain feature. You then will be given a single text that may or may not contain this feature. 

These features of text are normally identified by looking for specific words or patterns in the text. There are many features associated with a single token, but most features are not related to any token.

Your job is to identify if the last token of the example, which is marked between << and >>, is significantly related to the features shown and higlighted in the other texts. It is possible for a sentence to be related to the feature, but the last token not meaningfully related to the feature.

Return 1 only if the marked token is meaningfully related to the feature. Return 0 otherwise. You must return your response in a valid Python list. Do not return anything else besides a Python list.
"""

# feature 60, layer 11, 16k
CONTEXT_EXAMPLE_ONE_A = """Example contexts: 
 pppd: Connection terminated.\nI have contacted my ISP, <<Aqu>>iss and they are getting a engineer to come out to me, but as its a bank
%, 19 times out of 20.\nThe detailed findings from the survey can be found here: http://<<aquaculture>>.ca/files/
umi messenger bag / laptop carrier – (Elements style in copper color)\n4. <<Aqu>>amarine leverback earrings (rectangular cut) set in white gold\n5
BOS can also link to FAQs/tips provided by other sources such as the manufacturer or other websites.\n- <<Ane>>cdotal observations that might be of use to
 be organised for collection from our warehouse.\nCopyright © 2006-2018 - www.scientificwire.com<<Ay>>cliffe Fabrications
3, No. 860).\n- Simeon Edmunds. (1966). Spiritualism: A Critical Survey. <<Aquarian>> Press.\n

Text context:

 starting again. just trying to get my head round it!\nJan 2012 - Have ripped out all of the interior, just starting to <<aq>>
"""

CONTEXT_RESPONSE_ONE_A = "[1]"

CONTEXT_EXAMPLE_ONE_B = """Example contexts: 
 pppd: Connection terminated.\nI have contacted my ISP, <<Aqu>>iss and they are getting a engineer to come out to me, but as its a bank
%, 19 times out of 20.\nThe detailed findings from the survey can be found here: http://<<aquaculture>>.ca/files/
umi messenger bag / laptop carrier – (Elements style in copper color)\n4. <<Aqu>>amarine leverback earrings (rectangular cut) set in white gold\n5
BOS can also link to FAQs/tips provided by other sources such as the manufacturer or other websites.\n- <<Ane>>cdotal observations that might be of use to
 be organised for collection from our warehouse.\nCopyright © 2006-2018 - www.scientificwire.com<<Ay>>cliffe Fabrications
3, No. 860).\n- Simeon Edmunds. (1966). Spiritualism: A Critical Survey. <<Aquarian>> Press.\n

Text context:

 Akron, OH, at #17. Great work Fred! Oregon did incredibly well: Rogue #21, Hair of the Dog #24, <<Des>>
"""

CONTEXT_RESPONSE_ONE_B = "[0]"


# feature 40, layer 11, 16k
CONTEXT_EXAMPLE_TWO_A = """Example contexts: 
 ‘\nSuper_L,’ and <<save it>>. You can use “\nSuper_R‘ as well.There is a sense of relief and anticipation as the
 Resources tab, or you can edit the XML directly by clicking the file and choosing the strings.xml tab. After you <<save the file,>> the resources <<are automatically>>
 So, in the end, to finish the process, you should <<save changes>>. To do that, after you have chosen “Cash”, you should <<click>> on “
.d/httpd <<restart>>" or "/etc/init.d/httpd <<reload>>".\nIn "/etc/sudoers" there's a Cmnd_
<< update && sudo apt-get install>> frogr\nAlternatively pre-packed .deb files for Lucid and Maverick: –|Some editions, not listed here, are

Text context:

sy.list -O /etc/apt/sources.list.d/winehq.list\nsudo apt-get update\nsudo apt-get <<install>>
"""


CONTEXT_RESPONSE_TWO_A = "[1]"

CONTEXT_EXAMPLE_TWO_B = """Example contexts: 
 ‘\nSuper_L,’ and <<save it>>. You can use “\nSuper_R‘ as well.There is a sense of relief and anticipation as the
 Resources tab, or you can edit the XML directly by clicking the file and choosing the strings.xml tab. After you <<save the file,>> the resources <<are automatically>>
 So, in the end, to finish the process, you should <<save changes>>. To do that, after you have chosen “Cash”, you should <<click>> on “
.d/httpd <<restart>>" or "/etc/init.d/httpd <<reload>>".\nIn "/etc/sudoers" there's a Cmnd_
<< update && sudo apt-get install>> frogr\nAlternatively pre-packed .deb files for Lucid and Maverick: –|Some editions, not listed here, are

Text context:

> Add New Notification. Set a Title for the notification.\n- Now <<select>>
"""


CONTEXT_RESPONSE_TWO_B = "[0]"


GENERATION_CONTEXT_PROMPT = """Example contexts: 
{contexts}

Text context:

{examples}
"""







def prompt(examples, explanation,score=False, contexts=False):
    if not contexts:
        system_prompt = EXPLANATION_SYSTEM_PROMPT
        generation_prompt = GENERATION_EXPLANATION_PROMPT.format(
            explanation=explanation, examples=examples, score=score
    )
        if score:
            score1 = "Score: 0.9"
            score2 = "Score: 0.8"
            score3 = "Score: 0.7"
        else:
            score1 = ""
            score2 = ""
            score3 = ""

        defaults = [
            {"role": "user", "content": EXPLANATION_EXAMPLE_ONE_A.format(score=score1)},
            {"role": "assistant", "content": EXPLANATION_RESPONSE_ONE_A},
            {"role": "user", "content": EXPLANATION_EXAMPLE_ONE_B.format(score=score1)},
            {"role": "assistant", "content": EXPLANATION_RESPONSE_ONE_B},
            
            {"role": "user", "content": EXPLANATION_EXAMPLE_TWO_A.format(score=score2)},
            {"role": "assistant", "content": EXPLANATION_RESPONSE_TWO_A},
            {"role": "user", "content": EXPLANATION_EXAMPLE_TWO_B.format(score=score2)},
            {"role": "assistant", "content": EXPLANATION_RESPONSE_TWO_B},
            
            {"role": "user", "content": EXPLANATION_EXAMPLE_THREE_A.format(score=score3)},
            {"role": "assistant", "content": EXPLANATION_RESPONSE_THREE_A},
            {"role": "user", "content": EXPLANATION_EXAMPLE_THREE_B.format(score=score3)},
            {"role": "assistant", "content": EXPLANATION_RESPONSE_THREE_B},
        ]

        prompt = [
            {"role": "system", "content": system_prompt},
            *defaults,
            {"role": "user", "content": generation_prompt},
        ]

    else:
        generation_prompt = GENERATION_CONTEXT_PROMPT.format(
            contexts=explanation, examples=examples
        )

        defaults = [
            {"role": "user", "content": CONTEXT_EXAMPLE_ONE_A},
            {"role": "assistant", "content": CONTEXT_RESPONSE_ONE_A},
            {"role": "user", "content": CONTEXT_EXAMPLE_ONE_B},
            {"role": "assistant", "content": CONTEXT_RESPONSE_ONE_B},

            {"role": "user", "content": CONTEXT_EXAMPLE_TWO_A},
            {"role": "assistant", "content": CONTEXT_RESPONSE_TWO_A},
            {"role": "user", "content": CONTEXT_EXAMPLE_TWO_B},
            {"role": "assistant", "content": CONTEXT_RESPONSE_TWO_B},
        ]

        prompt = [
            {"role": "system", "content": CONTEXT_SYSTEM_PROMPT},
            *defaults,
            {"role": "user", "content": generation_prompt},
        ]

    return prompt

def finetuned_prompt(examples, explanation,soft=False,score=False,contexts=False):

    prompt = [
        {"role":"user", "content":f"\n[EXPLANATION]: {explanation}\n[SENTENCE]: {examples}"}
    ]
    return prompt

def simulation_prompt(examples, explanation):
    system_prompt = EXPLANATION_SYSTEM_PROMPT_SIM
    generation_prompt = GENERATION_EXPLANATION_PROMPT_SIM.format(
        explanation=explanation, examples=examples,score=""
    )

    defaults = [
        {"role": "user", "content": EXPLANATION_EXAMPLE_ONE_A_SIM},
        {"role": "assistant", "content": EXPLANATION_RESPONSE_ONE_A_SIM},
        
        {"role": "user", "content": EXPLANATION_EXAMPLE_TWO_A_SIM},
        {"role": "assistant", "content": EXPLANATION_RESPONSE_TWO_A_SIM},
        
        {"role": "user", "content": EXPLANATION_EXAMPLE_THREE_A_SIM},
        {"role": "assistant", "content": EXPLANATION_RESPONSE_THREE_A_SIM},
        ]

    prompt = [
        {"role": "system", "content": system_prompt},
        *defaults,
        {"role": "user", "content": generation_prompt},
    ]

    return prompt