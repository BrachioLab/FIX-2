
vanilla_baseline = """In addition to the answer, please provide a single paragraph under 200 characters explaining why you gave the answer you did."""

cot_baseline = """To come up with the correct answer, think step-by-step. You should walk through each step in your reasoning process and explain how you arrived at the answer. Describe your step-by-step reasoning in a single paragraph under 200 characters. This paragraph will serve as the explanation for your answer."""

socratic_baseline = """To come up with the correct answer, have a conversation with yourself. Pinpoint what you need to know, ask critical questions, and constantly challenge your understanding of the field. Describe this question-and-answer journey in a single paragraph under 200 characters. This paragraph will serve as the explanation for your answer."""

least_to_most_baseline = """To come up with the correct answer, determine all of the subquestions you must answer. Start with the easiest subquestion, answer it, and then use that subquestion and answer to tackle the next subquestion. Describe your subquestion decomposition and answers in a single paragraph under 200 characters. This paragraph will serve as the explanation for your answer."""

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
Explanation: <explanation, as desribed above>

Utterance: {}
"""

supernova_prompt = """What is the astrophysical classification of the following time series? Here are the possible labels you can use: type Ia supernova (SNIa), type II supernova (SNII), M-dwarf, eclipsing binary (EB), tidal disruption event (TDE), type Ibc supernova (SNIbc), or active galactic nuclei (AGN).
Each time series contains observations recorded over time, with each observation consisting of a timestamp, a wavelength (representing the spectral band), and a corresponding measurement value.

[BASELINE_PROMPT]

Your response should be 2 lines, formatted as follows:
Label: <supernova classification label>
Explanation: <explanation, as described above>

Here is the time series data for you to classify.
Time data: {time_data}
Wavelength data: {wv_data} 
Value data: {value_data} 
"""

sepsis_prompt = """What is the sepsis risk prediction for the following time series? Here are the possible labels you can use: Yes (the patient is at high risk of developing sepsis within 12 hours) or No (the patient is not at high risk of developing sepsis within 12 hours).
The time series consists of Electronic Health Record (EHR) data collected during the first 2 hours of the patient’s emergency department (ED) admission. Each entry includes a timestamp, the name of a measurement or medication, and its corresponding value.

[BASELINE_PROMPT]

Your response should be 2 lines, formatted as follows:
Label: <emotion label>
Explanation: <explanation, as described above>

Here is the text for you to classify.
Text: {}
"""
