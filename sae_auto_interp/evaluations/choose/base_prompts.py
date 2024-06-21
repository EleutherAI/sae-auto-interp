EXPLANATION_SYSTEM_POS = \
"We're studying neurons in a neural network."\
"Each neuron looks for some particular thing in a short document."\
"You will be given an explanation of what activates the neuron."\
"Sugest a short sentence that activates it. "\

EXPLANATION_SYSTEM_NEG = \
"We're studying neurons in a neural network."\
"Each neuron looks for some particular thing in a short document."\
"You will be given an explanation of what activates the neuron."\
"Sugest a short sentence that does not activate it. "\


first_document_user_pos = "Explanation: The neuron is looking for units of time.\n"
first_document_assistant_pos = "Benghazi, however, is still gripped by a revolutionary fervor less than two weeks since the rout of Gadhafi's forces and freeing of the east from his control.\n"

first_document_user_neg = "Explanation: The neuron is looking for units of time.\n"
first_document_assistant_neg = "Hooper said more application was needed by the batsmen. 'We need to turn 40s into hundreds,' he said. But Hooper said he was heartened\n"

second_document_user_pos = "Explanation: The neuron is looking for pronouns that are the active speakers in a sentence\n"
second_document_assistant_pos = "'It hasn't been a big month for sleeping,' he said. 'The launch has been an \n"

second_document_user_neg = "Explanation: The neuron is looking for pronouns that are the active speakers in a sentence\n"
second_document_assistant_neg = "Today in the anniversary of the first launch of the Space Shuttle\n"


FEW_SHOT_EXAMPLES_POS = {"example1":{"user": first_document_user_pos, "assistant": first_document_assistant_pos},"example2":{"user": second_document_user_pos, "assistant": second_document_assistant_pos}}

FEW_SHOT_EXAMPLES_NEG = {"example1":{"user": first_document_user_neg, "assistant": first_document_assistant_neg},"example2":{"user": second_document_user_neg, "assistant": second_document_assistant_neg}}