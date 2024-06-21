EXPLANATION_SYSTEM = \
"We're studying neurons in a neural network."\
"Each neuron looks for some particular thing in a short document."\
"You will be given the parts of the documents that activate the neuron."\
"Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for."\
"Be specific if you think it activates for a specific token. Don't list examples of words."\
"After each sentence you will get a list of tokens that activate the neuron and its activations. Activation values range from 0 to 10."\
"The higher the activation value, the stronger the match."\

first_neuron_user = "Neuron\n"\
"Document 1:\n"\
"the disparate units scattered across the east.\n"\
"Benghazi, however, is still gripped by a revolutionary fervor less than two weeks since the rout of Gadhafi's forces and freeing of the east from his control.\n"\
"Volunteers are still directing traffic and manning checkpoints into the early hours of the morning.\n"\
"Activating tokens: weeks (6).\n"\
"Document 2:\n"\
"the Planning Commission further argued.\n"\
"Meanwhile, agriculture minister Sharad Pawar on Monday evening recused himself from heading the Empowered Group of Ministers (EGoM) on telecom, three days after he replaced Pranab Mukherjee in the job.\n"\
"Pawar, who was scheduled to chair the first\n"\
"Activating tokens: days (6).\n"\
"Document 3:\n"\
"Rahm Emanuel, the former White House chief of staff who returned to the city to make his successful run for mayor.\n"\
"About six months after Daley left office, his wife, Maggie, died Thanksgiving Day after a long battle with breast cancer.\n"\
"Activating tokens: months (6.)\n"\
"Document 4:\n"\
"81 to lead Tennessee Tech in the first round of the Low Country Intercollegiate on Sunday at Moss Creek Plantation.\n"\
"March 24, 2012\n"\
"Carolina in my mind: Golden Eagles prep for Low Country Intercollegiate\n"\
"Less than a week after completing play at the 2012 Pinehurst Challenge, Tennessee\n"\
"Activating tokens: week (6).\n"\
"Document 5:\n"\
"some tributary of the Amazon, has offered their two bits' worth on the launch of Apple's iPhone, it comes as a bit of a surprise that al-Qaeda has dismally failed to contribute to the brouhaha.\n"\
"Why the iPhone is a success\n"\
"Two weeks after the iPhone virus "\
"Activating tokens: weeks (5).\n\n"
first_neuron_assistant="Explanation: The neuron is looking for units of time.\n\n"
second_neuron_user = "Neuron\n"\
"Document 1:\n"\
"'then fell away today,' he said."\
"Hooper said more application was needed by the batsmen. 'We need to turn 40s into hundreds,' he said. But Hooper said he was heartened\n"\
"that the West Indies had shown they were competitive during the first two Tests of a five-match series \n"\
"Activating tokens: he (4).\n"\
"Document 2:\n"\
"and-comers like Roll Call's Emily Heil and Politico's Kiki Ryan.\n"\
"Carlson said he starts the day at 5:30 a.m. to accommodate all the must-reads.\n"\
"'It hasn't been a big month for sleeping,' he said. 'The launch has been an \n"\
"Activating tokens: he (3).\n"\
"Document 3:\n"\
"four to six weeks,' Mr Newell said.\n"\
"'Ian put a couple of papers out a week virtually single-handed.'\n"\
"Mrs Fell yesterday said after his family and friends, newspapers were her husband's great passion.\n"\
"'He spent 40 years in newspapers,' she said. 'In the early days he\n"\
"Activating tokens: she (3).\n\n"
second_neuron_assistant="Explanation: The neuron is looking for pronouns that are the active speakers in a sentence\n\n"\

FEW_SHOT_EXAMPLES = {"example1":{"user": first_neuron_user, "assistant": first_neuron_assistant},"example2":{"user": second_neuron_user, "assistant": second_neuron_assistant}}

