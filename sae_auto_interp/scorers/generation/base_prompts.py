EXPLANATION_SYSTEM_POS = \
"We're studying neurons in a neural network."\
"Each neuron looks for some particular thing in a short document."\
"You will be given an explanation of what activates the neuron."\
"Sugest a 5 sentences that activate it. "\

EXPLANATION_SYSTEM_NEG = \
"We're studying neurons in a neural network."\
"Each neuron looks for some particular thing in a short document."\
"You will be given an explanation of what activates the neuron."\
"Sugest a 5 sentences that do not activate it. "\


first_document_user_pos = "Explanation: The neuron is looking for units of time.\n"
first_document_assistant_pos = "1. Benghazi, however, is still gripped by a revolutionary fervor less than two weeks since the rout of Gadhafi's forces and freeing of the east from his control.\n"\
"2. Volunteers are still directing traffic and manning checkpoints into the early hours of the morning.\n"\
"3. Less than one day ago, there was an eclipse.\n"\
"4. Every year, in the 5th of October, there is a parade.\n"\
"5. A new video game will be released in the next few months.\n"
second_document_user_pos = "Explanation: The neuron is looking for pronouns that are the active speakers in a sentence\n"
second_document_assistant_pos = "1. 'It hasn't been a big month for sleeping,' he said.\n"\
"2. 'Its possible that today rains' she said.\n"\
"3. 'I am going to the store' he said. 'I really like their chocolate brand'.\n"\
"4. The day passed and nothing happened. 'I am bored' he said.\n"\
"5. 'I have a bad feeling about this' she said. 'I think we should leave'.\n"

first_document_user_neg = "Explanation: The neuron is looking for units of time.\n"
first_document_assistant_neg = "1. Hooper said more application was needed by the batsmen.\n"
"2. That sofa is very comfortable. I wish I had it in my home.\n"
"3. The new restaurant is very popular. Their new chef won a Michelin star.\n"
"4. I got up and started running after I heard the explosion.\n"
"5. This movie is very good. I think it will win an Oscar.\n"


second_document_user_neg = "Explanation: The neuron is looking for pronouns that are the active speakers in a sentence\n"
second_document_assistant_neg = "1. Today in the anniversary of the first launch of the Space Shuttle\n"
"2. The new book is very interesting. I think it will be a best seller.\n"
"3. 'I am going to the store' John said. 'I really like their chocolate brand'.\n"
"4. The day passed and nothing happened. 'I am bored' Maria said.\n"
"5. 'I have a bad feeling about this' Michael said. 'I think we should leave'.\n"


FEW_SHOT_EXAMPLES_POS = {"example1":{"user": first_document_user_pos, "assistant": first_document_assistant_pos},"example2":{"user": second_document_user_pos, "assistant": second_document_assistant_pos}}

FEW_SHOT_EXAMPLES_NEG = {"example1":{"user": first_document_user_neg, "assistant": first_document_assistant_neg},"example2":{"user": second_document_user_neg, "assistant": second_document_assistant_neg}}