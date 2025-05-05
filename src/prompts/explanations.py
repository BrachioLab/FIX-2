vanilla_baseline = """In addition to the answer, please provide a single paragraph under 200 words explaining why you gave the answer you did."""

cot_baseline = """To come up with the correct answer, think step-by-step. You should walk through each step in your reasoning process and explain how you arrived at the answer. Describe your step-by-step reasoning in a single paragraph under 200 words. This paragraph will serve as the explanation for your answer."""

socratic_baseline = """To come up with the correct answer, have a conversation with yourself. Pinpoint what you need to know, ask critical questions, and constantly challenge your understanding of the field. Describe this question-and-answer journey in a single paragraph under 200 words. This paragraph will serve as the explanation for your answer."""

least_to_most_baseline = """To come up with the correct answer, determine all of the subquestions you must answer. Start with the easiest subquestion, answer it, and then use that subquestion and answer to tackle the next subquestion. Describe your subquestion decomposition and answers in a single paragraph under 200 words. This paragraph will serve as the explanation for your answer."""


#-----------------------------------------------------------

emotion_prompt = """What is the emotion of the following text? Here are the possible labels you could use: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, or neutral.

[BASELINE_PROMPT]

Your response should be 2 lines, formatted as follows:
Label: <emotion label>
Explanation: <explanation, as described above>

Here is the text for you to classify.
Text: {}
"""

politeness_prompt = """What is the politeness of the following utterance on a scale of 1-5? Use the following scale:
1: extremely rude
2: somewhat rude
3: neutral
4: somewhat polite
5: extremely polite

[BASELINE_PROMPT]

Your response should be 2 lines, formatted as follows:
Rating: <politeness rating>
Explanation: <explanation, as described above>

Utterance: {}
"""


cardiac_prompt = """You are a medical expert specializing in cardiac arrest prediction. 
You will be provided with time-series Electrocardiogram (ECG) data from the first {} of an ECG monitoring period during a patient's ICU stay. Each entry consists of a measurement value at that timestamp. The timestamps start at time {} and end at time {}. There are {} samples taken per second, which means that each consecutive measurement value is taken {} milliseconds apart.

Your task is to determine whether this patient is at high risk of experiencing cardiac arrest within the next {} minutes.
Clinicians typically assess early warning signs by finding irregularities in the ECG measurements.
[BASELINE_PROMPT]
Focus on the features of the data you used to make your yes or no binary classification. 
Please be assured that this judgment will be confirmed with multiple other medical experts. Please provide your best judgment without worrying about not providing the perfect answer.

Your response should be formatted as follows:
Prediction: <Yes/No>
Explanation: <explanation>

Here is the ECG data for you to analyze:
{}
"""