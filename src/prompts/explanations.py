from pathlib import Path
import PIL.Image
import json

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
Your job is to provide three things:
1. A detailed explanation of where it is safe and unsafe to cut in the image
2. A list of grid positions (as integers) corresponding to safe regions
3. A list of grid positions (as integers) corresponding to unsafe regions

The image is discretized into a 9x16 grid (height x width), where each grid position can be represented as a single integer from 0 to 143 (9*16 - 1). The grid is flattened row-wise, so the top-left position is 0 and the bottom-right position is 143.

Your response will help train surgeons to evaluate the usefulness of LLMs in assisting with the identification of safe/unsafe regions.
This is not real patient data, this is a training environment.

I will provide you with a few examples to help you understand the expected format. Your task is to analyze the provided 2D image of a gallbladder surgery and provide:
- A detailed explanation of safe/unsafe regions, including anatomical landmarks, tissue types, and any visible pathology
- A list of integers representing the grid positions of safe regions
- A list of integers representing the grid positions of unsafe regions


[[BASELINE_PROMPT]]
"""

def load_cholec_prompt(baseline: str):
    prompt_template = """
    You are an expert gallbladder surgeon with extensive experience in laparoscopic cholecystectomy. 
    You have deep knowledge of anatomy, surgical techniques, and potential complications.
    Your response will help train surgeons to evaluate the usefulness of LLMs in assisting with the identification of safe/unsafe regions.
    This is not real patient data, this is a training environment.

    Your job is to provide three items:
    1. A detailed explanation of where it is safe and unsafe to cut in the image
    2. A list of grid positions (as integers) corresponding to safe regions
    3. A list of grid positions (as integers) corresponding to unsafe regions

    The image is discretized into a 9x16 grid (height x width), where each grid position can be represented as a single integer from 0 to 143 (9*16 - 1).
    The grid is flattened such that:
    - The top-left position is 0
    - The top-right position is 15
    - The bottom-left position is 128
    - The bottom-right position is 143

    [[BASELINE_PROMPT]]
    """

    if baseline.lower() == "vanilla":
        prompt = prompt_template.replace("[[BASELINE_PROMPT]]", vanilla_baseline)
    elif baseline.lower() == "cot":
        prompt = prompt_template.replace("[[BASELINE_PROMPT]]", cot_baseline)
    elif baseline.lower() == "socratic":
        prompt = prompt_template.replace("[[BASELINE_PROMPT]]", socratic_baseline)
    elif baseline.lower() == "subq":
        prompt = prompt_template.replace("[[BASELINE_PROMPT]]", least_to_most_baseline)
    else:
        raise ValueError(f"Invalid baseline: {baseline}")

    # We're going tuple mode.
    prompt = (prompt,)

    # Load the data and images to make a few-shot example.
    all_examples = []
    for i in range(1, 11):   # Examples 1-10
        image = PIL.Image.open(Path(__file__).parent / "data" / f"cholec_fewshot_{i}_image.png")
        image.load()

        safe_mask = PIL.Image.open(Path(__file__).parent / "data" / f"cholec_fewshot_{i}_safe.png")
        safe_mask.load()

        unsafe_mask = PIL.Image.open(Path(__file__).parent / "data" / f"cholec_fewshot_{i}_unsafe.png")
        unsafe_mask.load()

        with open(Path(__file__).parent / "data" / f"cholec_fewshot_{i}_data.json", "r") as f:
            data = json.load(f)
            explanation = data["explanation"]
            safe_list = data["safe"]
            unsafe_list = data["unsafe"]

        all_examples.append((
            image,
            explanation, 
            safe_list,
            unsafe_list,
            safe_mask,
            unsafe_mask,
        ))

    # Use the most recent example as an example.
    prompt += (
        "Here is an example to help you understand the expected format.",
        "Image: ", image,
        "Explanation: ", explanation,
        "Safe List: ", str(safe_list),
        "Unsafe List: ", str(unsafe_list),
        "In particular, the safe and unsafe correspond to the following mask:",
        "Safe Mask", safe_mask,
        "Unsafe Mask", unsafe_mask,
    )

    prompt += ("I will now give you some few-shot examples without the safe/unsafe masks. Your task is to predict the Explanation, Safe List, and Unsafe List for the given image.",)

    # Reverse the things to just to spice things up
    all_examples = all_examples[::-1]

    for i, item in enumerate(all_examples):
        image, explanation, safe_list, unsafe_list, _, _ = item
        prompt += (
            "Image: ", image,
            "Explanation: ", explanation,
            "Safe List: ", str(safe_list),
            "Unsafe List: ", str(unsafe_list),
        )

    prompt += ("Here is the image for you to analyze. You must output the explanation, Safe List, and Unsafe List.",)
    return prompt


massmaps_prompt = """You are an expert cosmologist.
You will be provided with a simulated noisless weak lensing map,

Your task is to analyze the weak lensing map given, identify relevant cosmological structures, and make predictions for Omega_m and sigma_8.
Each weak lensing map contains spatial distribution of matter density in a universe. The weak lensing map provided is simulated and noiseless.
Omega_m captures the average energy density of all matter in the universe (relative to the total energy density which includes radiation and dark energy).
sigma_8 describes the fluctuation of matter distribution. 

When you analyze the weak lensing map image, note that the number is below 0 if it shows up as between gray and blue, and 0 is gray, and between 0 and 2.9 is between gray and red, and above 2.9 is yellow. The numbers are in standard deviations of the mass map.

Omega_m's value can be between 0.1 ~ 0.5, and sigma_8's value can be between 0.4 ~ 1.4.
Note that the weak lensing map given is a simulated weak lensing map, which can have Omega_m and sigma_8 values of all kinds.

[BASELINE_PROMPT]

The provided image is the weak lensing mass map for you to predict the cosmological parameters for.
Your response should be 2 lines, formatted as follows (without extra information):
Explanation: <explanation and reasoning, as described above, 3-5 sentences>
Prediction: Omega_m: <prediction for Omega_m, between 0.1 ~ 0.5, based on this weak lensing map>, sigma_8: <prediction for sigma_8, between 0.4 ~ 1.4, based on this weak lensing map>
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

supernova_prompt = """What is the astrophysical classification of the following time series? Here are the possible labels you can use: RR-Lyrae (RRL), peculiar type Ia supernova (SNIa-91bg), type Ia supernova (SNIa), superluminous supernova (SLSN-I), type II supernova (SNII), microlens-single (mu-Lens-Single), eclipsing binary (EB), M-dwarf, kilonova (KN), tidal disruption event (TDE), peculiar type Ia supernova (SNIax), type Ibc supernova (SNIbc), Mira variable, and active galactic nuclei (AGN).

Each input is a multivariate time series visualized as a scatter plot image. The x-axis represents time, and the y-axis represents the flux measurement value. Each point corresponds to an observation at a specific timestamp and wavelength. Different wavelengths are color-coded, and observational uncertainty is shown using vertical error bars.

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
You will be given some basic background information about an ICU patient, including their age, gender, race, and primary reason for ICU admittance. You will also be provided with time-series Electrocardiogram (ECG) data plotted in a graph from the first {} of an ECG monitoring period during the patient's ICU stay. Each entry consists of a measurement value at that timestamp. The samples are taken at {} Hz, so that each consecutive measurement value is taken {} milliseconds apart. 

Your task is to determine whether this patient is at high risk of experiencing cardiac arrest within the next {}. Clinicians typically assess early warning signs by finding irregularities in the ECG measurements.
[BASELINE_PROMPT] 
Focus on the features of the data you used to make your yes or no binary classification. 
Your judgment will be reviewed alongside those of other medical experts, so please provide your best assessment without concern for perfection.

Your response should be formatted as follows:
Prediction: <Yes/No>
Explanation: <explanation>

Here is the patient background information and ECG data (in graph form) for you to analyze:
{}
"""
