import pandas as pd
import numpy as np
from datasets import load_dataset
import openai
from openai import OpenAI
import time
from tqdm import tqdm
import json
from fuzzywuzzy import fuzz
import anthropic
import google.generativeai as genai


from prompts.claim_decomposition import decomposition_emotion
from prompts.relevance_filtering import relevance_emotion
from prompts.expert_alignment import alignment_emotion
from prompts.explanations import vanilla_baseline, cot_baseline, socratic_baseline, least_to_most_baseline, emotion_prompt

from diskcache import Cache
cache = Cache("/shared_data0/shreyah/llm_cache")

prompt_dict = {"vanilla": vanilla_baseline,
               "cot": cot_baseline,
               "socratic": socratic_baseline,
               "subq": least_to_most_baseline}
categories_list = [
    "Valence",
    "Arousal",
    "Emotion Words & Emojis",
    "Expressive Punctuation",
    "Humor/Laughter Markers",
    "Confusion Phrases",
    "Curiosity Questions",
    "Surprise Exclamations",
    "Threat/Worry Language",
    "Loss or Let-Down Words",
    "Other-Blame Statements",
    "Self-Blame & Apologies",
    "Aversion Terms",
    "Praise & Compliments",
    "Gratitude Expressions",
    "Affection & Care Words",
    "Self-Credit Statements",
    "Relief Indicators"
]

emotion_labels = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral"
}

class EmotionExample:
    def __init__(self, text, ground_truth, llm_label, llm_explanation):
        self.text = text
        self.ground_truth = ground_truth
        self.llm_label = llm_label
        self.llm_explanation = llm_explanation
        self.accuracy = 0.0
        self.claims = []
        self.relevant_claims = []
        self.alignment_scores = []
        self.alignment_categories = []
        self.alignment_reasonings = []
        self.final_alignment_score = 0.0

    def print(self, verbose=False):
        print("Text: ", self.text)
        print("Ground Truth: ", self.ground_truth)
        print("LLM Label: ", self.llm_label)
        print("LLM Explanation: ", self.llm_explanation)
        print("Claims: ", self.claims)
        print("Relevant Claims: ", self.relevant_claims)
        print("Alignment Scores: ", self.alignment_scores)
        
        print("Final Alignment Score: ", self.final_alignment_score)
        if(verbose):
            print("Alignment Categories: ", self.alignment_categories)
            print("Alignment Reasonings: ", self.alignment_reasonings)
    
    def to_dict(self):
        return {
            'text': self.text,
            'ground_truth': self.ground_truth,
            'llm_label': self.llm_label,
            'llm_explanation': self.llm_explanation,
            'accuracy': self.accuracy,
            'claims': self.claims,
            'relevant_claims': self.relevant_claims,
            'alignment_scores': self.alignment_scores,
            'alignment_categories': self.alignment_categories,
            'alignment_reasonings': self.alignment_reasonings,
            'final_alignment_score': self.final_alignment_score
        }
    
@cache.memoize()
def query_anthropic(prompt, model="claude-3-5-sonnet-latest"):
    with open("../Anthropic_API_KEY.txt", "r") as file:
        api_key = file.read()
    client = anthropic.Anthropic(api_key=api_key)

    num_tries = 0
    for i in range(3):
        try:
            response = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ])
            return response.content[0].text
        except Exception as e:
            num_tries += 1
            print("Try {}; Error: {}".format(str(num_tries), str(e)))     
            time.sleep(5)


@cache.memoize()
def query_gemini(prompt, model="gemini-2.0-flash"):
    with open("../Google_API_KEY.txt", "r") as file:
        api_key = file.read()
    genai.configure(api_key=api_key)
    
    num_tries = 0
    for i in range(3):
        try:
            model_obj = genai.GenerativeModel(model)
            response = model_obj.generate_content(prompt)
            return response.text
        except Exception as e:
            num_tries += 1
            print("Try {}; Error: {}".format(str(num_tries), str(e)))     
            time.sleep(5)
    return "ERROR"
    

@cache.memoize()
def query_openai(prompt, model="gpt-4o"):
    with open("../API_KEY.txt", "r") as file:
        api_key = file.read()
    client = OpenAI(api_key=api_key)

    num_tries = 0
    for i in range(3):
        try:
            response = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
                model=model,
            )
            return response.choices[0].message.content
        except Exception as e:
            num_tries += 1
            print("Try {}; Error: {}".format(str(num_tries), str(e)))     
            time.sleep(5)
    return "ERROR"

def get_llm_generated_answer(text: str, baseline: str = "vanilla", model = "gpt-4o"):
    prompt = emotion_prompt.replace("[BASELINE_PROMPT", prompt_dict[baseline]).format(text)

    if("gpt" in model or "o1" in model):
        response = query_openai(prompt, model=model).replace("\n\n", "\n")
    elif("claude" in model):
        response = query_anthropic(prompt, model=model).replace("\n\n", "\n")
    elif("gemini" in model):
        response = query_gemini(prompt, model=model).replace("\n\n", "\n")
    else:
        print("ERROR: Model not supported")
        return None

    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response_split = [e for e in response.split("\n") if (e != '' and e.split()[0] in ['Label:', 'Explanation:'])]
    llm_label = response_split[0].split("Label: ")[1].strip().lower()
    explanation = response_split[1].split("Explanation: ")[1].strip()
    try:
        assert(len(explanation) > 10)
        return llm_label, explanation
    except:
        print("ERROR: LLM generated answer is not valid")
        print(response)
        return None, None


def isolate_individual_features(explanation: str):
    prompt = decomposition_emotion.format(explanation)
    response = query_openai(prompt).replace("\n\n", "\n")
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("OUTPUT:", "").strip()
    claims = response.split("\n")
    return claims

def is_claim_relevant(text: str, rating: str, claim: str):
    prompt = relevance_emotion.format(text, rating, claim)
    response = query_openai(prompt).replace("\n\n", "\n")
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    try:
        response = response.replace("Relevance:", "").strip()
        response = response.split("\n")
        relevance = response[0].strip()
        reasoning = response[1].replace("Reasoning:", "").strip()
        assert(relevance in ["Yes", "No"])
        assert(len(reasoning) > 10)
    except:
        print("ERROR: Could not determine relevance")
        print(response)
        return None, None
    return relevance, reasoning


def distill_relevant_features(example: EmotionExample):
    relevant_claims = []
    for claim in tqdm(example.claims):
        relevance, reasoning = is_claim_relevant(example.text, example.llm_label, claim)
        if relevance is None:
            continue
        if relevance == "Yes":
            relevant_claims.append(claim)
    return relevant_claims

def calculate_expert_alignment_score(claim: str):
    prompt = alignment_emotion.format(claim)
    response = query_openai(prompt).replace("\n\n", "\n")
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("Category:", "").strip()
    response = response.split("\n")
    response = [r for r in response if r.strip() != ""]
    category = response[0].strip().replace("‑", "-")
    alignment_score = response[1].replace("Category Alignment Rating:", "").strip()
    reasoning = response[2].replace("Reasoning:", "").strip()
    try:
        alignment_score = float(alignment_score)
        assert(len(category) > 5)
        for c in categories_list:
            if fuzz.ratio(c.lower(), category.lower()) > 90:
                category = c
                break
        if(category not in categories_list): category = None
        assert(len(reasoning) > 10)
    except:
        print("ERROR: Issue with alignment score parsing")
        print(response)
        alignment_score = 0.0
    return category, alignment_score, reasoning

def load_emotion_data():
    emotion_data =  load_dataset("BrachioLab/emotion")
    emotion_data = emotion_data['train'].to_pandas()
    emotion_data['labels'] = emotion_data['labels'].apply(lambda x: [int(i) for i in x])
    emotion_data = emotion_data[emotion_data['text'].apply(lambda x: len(x) > 20)]
    
    #sample 4 examples from each label
    labels = [[x] for x in range(28)]
    emotion_data_sampled = pd.DataFrame()
    for l in labels:
        label_sample = emotion_data[emotion_data['labels'].apply(lambda x: x == l)].sample(4, random_state=11)
        emotion_data_sampled = pd.concat([emotion_data_sampled, label_sample])
    emotion_data = emotion_data_sampled.reset_index(drop=True)
    return emotion_data


def run_pipeline(emotion_data, baseline="vanilla", model="gpt-4o"):
    emotion_examples = []
    for idx,row in tqdm(emotion_data.iterrows()):
        label, explanation = get_llm_generated_answer(row['text'], baseline, model)
        if label is None:
            continue
        emotion_examples.append(EmotionExample(
            text=row['text'],
            ground_truth=emotion_labels[int(row['labels'][0])],
            llm_label=label,
            llm_explanation=explanation
        ))
    
    for example in emotion_examples:
        example.accuracy = int(example.ground_truth == example.llm_label)

    for example in emotion_examples:
        claims = isolate_individual_features(example.llm_explanation)
        if claims is None:
            continue
        example.claims = [claim.strip() for claim in claims]

    for example in emotion_examples:
        relevant_claims = distill_relevant_features(example)
        example.relevant_claims = relevant_claims

    for example in emotion_examples:
        alignment_scores = []
        alignment_categories = []
        alignment_reasonings = []
        for claim in tqdm(example.relevant_claims):
            category, alignment_score, reasoning = calculate_expert_alignment_score(claim)
            if category is None:
                continue
            alignment_scores.append(alignment_score)
            alignment_categories.append(category)
            alignment_reasonings.append(reasoning)
        example.alignment_scores = alignment_scores
        example.alignment_categories = alignment_categories
        example.final_alignment_score = np.sum(alignment_scores)/len(example.claims)
        example.alignment_reasonings = alignment_reasonings

        
    data_to_save = [example.to_dict() for example in emotion_examples]
    with open("../results/{}/emotion_{}.json".format(baseline, model), 'w') as f:
        json.dump(data_to_save, f, indent=4)

if __name__ == "__main__":
    emotion_data = load_emotion_data()

    #model = "gemini-2.0-flash"
    run_pipeline(emotion_data, baseline="vanilla", model="gemini-2.0-flash")
    run_pipeline(emotion_data, baseline="cot", model="gemini-2.0-flash")
    run_pipeline(emotion_data, baseline="socratic", model="gemini-2.0-flash")
    run_pipeline(emotion_data, baseline="subq", model="gemini-2.0-flash")

    #model = "o1"
    run_pipeline(emotion_data, baseline="vanilla", model="gpt-4o")
    run_pipeline(emotion_data, baseline="cot", model="gpt-4o")
    run_pipeline(emotion_data, baseline="socratic", model="gpt-4o")
    run_pipeline(emotion_data, baseline="subq", model="gpt-4o")

    #model = "claude-3-5-sonnet-latest"
    run_pipeline(emotion_data, baseline="vanilla", model="claude-3-5-sonnet-latest")
    run_pipeline(emotion_data, baseline="cot", model="claude-3-5-sonnet-latest")
    run_pipeline(emotion_data, baseline="socratic", model="claude-3-5-sonnet-latest")
    run_pipeline(emotion_data, baseline="subq", model="claude-3-5-sonnet-latest")
