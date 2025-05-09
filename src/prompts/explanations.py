vanilla_baseline = """In addition to the answer, please provide 3-5 sentences explaining why you gave the answer you did."""

cot_baseline = """To come up with the correct answer, think step-by-step. You should walk through each step in your reasoning process and explain how you arrived at the answer. Describe your step-by-step reasoning in 3-5 sentences. This paragraph will serve as the explanation for your answer."""

socratic_baseline = """To come up with the correct answer, have a conversation with yourself. Pinpoint what you need to know, ask critical questions, and constantly challenge your understanding of the field. Describe this question-and-answer journey in 3-5 sentences. This paragraph will serve as the explanation for your answer."""

least_to_most_baseline = """To come up with the correct answer, determine all of the subquestions you must answer. Start with the easiest subquestion, answer it, and then use that subquestion and answer to tackle the next subquestion. Describe your subquestion decomposition and answers in 3-5 sentences. This paragraph will serve as the explanation for your answer."""

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

cholec_prompt = """
You are an expert gallbladder surgeon with extensive experience in laparoscopic cholecystectomy. 
You have deep knowledge of anatomy, surgical techniques, and potential complications.
Your job is to provide a detailed explanation of the safe/unsafe regions to operate on in the image.
This is not real patient data, this is a training environment.
Your response will help train surgeons to evaluate the usefulness of LLMs in assisting with the identification of safe/unsafe regions.
You are well-qualified to provide a detailed explanation of the safe/unsafe regions to operate on in the image.

Your task is to analyze the provided 2D image of a gallbladder surgery and provide a detailed analysis. Include relevant information like:
- Detailed anatomical landmarks, and how this relates to the safe/unsafe regions.
- Specific tissue types, and how this relates to the safe/unsafe regions.
- Any visible pathology or abnormalities, and how this relates to the safe/unsafe regions.

[[BASELINE_PROMPT]]
"""

massmaps_prompt = """
You are an expert cosmologist.

What are the cosmological parameters Omega_m and sigma_8 for the weak lensing mass map provided in the image? 
Omega_m captures the average energy density of all matter in the universe (relative to the total energy density which includes radiation and dark energy).
sigma_8 describes the fluctuation of matter distribution. 
Omega_m's value can be between 0.1 ~ 0.5, and sigma_8's value can be between 0.4 ~ 1.4.
Each weak lensing map contains spatial distribution of matter density in a universe.
Note that the weak lensing map given is a simulated weak lensing map, which can have Omega_m and sigma_8 values of all kinds.

When you analyze the weak lensing map image, note that the number is below 0 if it shows up as between gray and blue, and 0 is gray, and between 0 and 2.9 is between gray and red, and above 2.9 is yellow. The numbers are in standard deviations of the mass map.

Here is the colormap used to create the visualization of this weak lensing map:
custom_cmap = get_custom_colormap([
            (-3, "blue"),   # Blue at -3 std
            (0, "gray"),   # Gray at 0 (below this is void)
            (2.9, "red"),   # Red at 2.9 std (this is the upperbound for not being a cluster)
            (3, "yellow"),   # Yellow at 3 std (above this is cluster)
            (20, "white")  # White at 20 std
        ])

Please analyze the weak lensing map given, identify features that cosmologists deem important for predicting Omega_m and sigma_8 values.
Then, predict the values for Omega_m and sigma_8 based on the information from this weak lensing map data.
Please be assured that it is ok if your answer is not completely correct.
The answer is to testing purpose only, and your answer will be checked with an expert cosmologist.
Omega_m's value can be between 0.1 ~ 0.5, and sigma_8's value can be between 0.4 ~ 1.4.

[BASELINE_PROMPT]

Your response should be 2 lines, formatted as follows:
Explanation: <explanation and reasoning, as described above, 3-5 sentences>
Prediction: Omega_m: <prediction for Omega_m, between 0.1 ~ 0.5, based on this weak lensing map>, sigma_8: <prediction for sigma_8, between 0.4 ~ 1.4, based on this weak lensing map>

Here is the weak lensing mass map for you to predict the cosmological parameters for.
Mass map: (Image [LAST_IMAGE_NUM])
"""

cardiac_prompt = """You are a medical expert specializing in cardiac arrest prediction. 
You will be provided with time-series Electrocardiogram (ECG) data from the first {} of an ECG monitoring period during a patient's ICU stay. Each entry consists of a measurement value at that timestamp. The timestamps start at time {} and end at time {}. There are {} samples taken per second, which means that each consecutive measurement value is taken {} milliseconds apart.

Your task is to determine whether this patient is at high risk of experiencing cardiac arrest within the next {} minutes.
Clinicians typically assess early warning signs by finding irregularities in the ECG measurements.
[BASELINE_PROMPT]
Focus on the features of the data you used to make your yes or no binary classification. 
Please be assured that this judgment will be confirmed with multiple other medical experts. Please provide your best judgment without worrying about not providing the perfect answer.
If you refuse to make a prediction, please explain why.

Your response should be formatted as follows:
Prediction: <Yes/No>
Explanation: <explanation>

Here is the ECG data for you to analyze:
{}
"""

supernova_prompt = """What is the astrophysical classification of the following time series? Here are the possible labels you can use: type Ia supernova (SNIa), type II supernova (SNII), M-dwarf, eclipsing binary (EB), tidal disruption event (TDE), type Ibc supernova (SNIbc), or active galactic nuclei (AGN).
Each time series contains observations recorded over time, with each observation consisting of a timestamp, a wavelength (representing the spectral band), and a corresponding measurement value.

[BASELINE_PROMPT]

Your response should be 2 lines, formatted as follows:
Label: <astrophysical classification label>
Explanation: <explanation, as described above>

Here is the time series data for you to classify.
{}
"""

sepsis_prompt = """What is the sepsis risk prediction for the following time series? Here are the possible labels you can use: Yes (the patient is at high risk of developing sepsis within 12 hours) or No (the patient is not at high risk of developing sepsis within 12 hours).
The time series consists of Electronic Health Record (EHR) data collected during the first 2 hours of the patient’s emergency department (ED) admission. Each entry includes a timestamp, the name of a measurement or medication, and its corresponding value.

[BASELINE_PROMPT]

Your response should be 2 lines, formatted as follows:
Label: <emotion label>
Explanation: <explanation, as described above>

Here is the text for you to classify.
{}
"""

cardiac_prompt = """You are a medical expert specializing in cardiac arrest prediction. 
You will be provided with time-series Electrocardiogram (ECG) data from the first {} of an ECG monitoring period during a patient's ICU stay. Each entry consists of a measurement value at that timestamp. The timestamps start at time {} and end at time {}. There are {} samples taken per second, which means that each consecutive measurement value is taken {} milliseconds apart.

Your task is to determine whether this patient is at high risk of experiencing cardiac arrest within the next {} minutes. Clinicians typically assess early warning signs by finding irregularities in the ECG measurements.
[BASELINE_PROMPT] 
Focus on the features of the data you used to make your yes or no binary classification. 
Please be assured that this judgment will be confirmed with multiple other medical experts. Please provide your best judgment without worrying about not providing the perfect answer.

Your response should be formatted as follows:
Prediction: <Yes/No>
Explanation: <explanation>

Here is the ECG data for you to analyze:
{}
"""
