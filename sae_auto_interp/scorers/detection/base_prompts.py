EXPLANATION_SYSTEM = \
"We're studying neurons in a neural network."\
"Each neuron looks for some particular thing in a short document."\
"You will be given an explanation of what activates the neuron and a document."\
"Look at the parts of the document the neuron should activate. "\
"If it activates say 'Yes', otherwise say 'No'.\n\n"

first_document_user = "Explanation: The neuron is looking for units of time.\n"\
"Document\n"\
"the disparate units scattered across the east.\n"\
"Benghazi, however, is still gripped by a revolutionary fervor less than two weeks since the rout of Gadhafi's forces and freeing of the east from his control.\n"\
"Volunteers are still directing traffic and manning checkpoints into the early hours of the morning.\n"
first_document_assistant = "Yes.\n"
second_document_user = "Explanation: The neuron is looking for units of time.\n"\
"Document\n"\
"81 to lead Tennessee Tech in the first round of the Low Country Intercollegiate on Sunday at Moss Creek Plantation.\n"\
"March 24, 2012\n"\
"'then fell away today,' he said."\
"Hooper said more application was needed by the batsmen. 'We need to turn 40s into hundreds,' he said. But Hooper said he was heartened\n"\
"that the West Indies had shown they were competitive during the first two Tests of a five-match series \n"
second_document_assistant = "No .\n"
third_document_user = "Explanation: The neuron is looking for pronouns that are the active speakers in a sentence\n"\
"Document\n"\
"and-comers like Roll Call's Emily Heil and Politico's Kiki Ryan.\n"\
"Carlson said he starts the day at 5:30 a.m. to accommodate all the must-reads.\n"\
"'It hasn't been a big month for sleeping,' he said. 'The launch has been an \n"
third_document_assistant = "Yes.\n"
fourth_document_user = "Explanation: The neuron is looking for pronouns that are the active speakers in a sentence\n"\
"Document\n"\
"some tributary of the Amazon, has offered their two bits' worth on the launch of Apple's iPhone, it comes as a bit of a surprise that al-Qaeda has dismally failed to contribute to the brouhaha.\n"\
"Why the iPhone is a success\n"\
"Two weeks after the iPhone virus "
fourth_document_assistant = "No.\n"

FEW_SHOT_EXAMPLES = {"example1":{"user": first_document_user, "assistant": first_document_assistant},"example2":{"user": second_document_user, "assistant": second_document_assistant},"example3":{"user": third_document_user, "assistant": third_document_assistant},"example4":{"user": fourth_document_user, "assistant": fourth_document_assistant}}
