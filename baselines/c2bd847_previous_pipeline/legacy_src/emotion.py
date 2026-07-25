import pandas as pd
import numpy as np
from datasets import load_dataset
import openai
from openai import OpenAI
import time
from tqdm import tqdm

import re
import json
from fuzzywuzzy import fuzz
import anthropic
import google.generativeai as genai


from prompts.claim_decomposition import decomposition_emotion
from prompts.relevance_filtering import relevance_emotion
from prompts.expert_alignment import alignment_emotion
from prompts.explanations import vanilla_baseline, cot_baseline, socratic_baseline, least_to_most_baseline, emotion_prompt

from diskcache import Cache
# cache = Cache("/shared_data0/shreyah/llm_cache")
from pathlib import Path

from llms import load_model

base_dir = Path(__file__).parent
cache_path = base_dir / ".." / ".." / "llm_cache"
cache = Cache(cache_path)

default_model = "gpt-4o"


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
    return load_model(model)(prompt)


@cache.memoize()
def query_gemini(prompt, model="gemini-2.0-flash"):
    return load_model(model)(prompt)
    

@cache.memoize()
def query_openai(prompt, model="gpt-4o"):
    return load_model(model)(prompt)

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


def isolate_individual_features(explanation: str, model: str = default_model):
    
    prompt = decomposition_emotion.format(explanation)
    
    if model == default_model:
        response = query_openai(prompt).replace("\n\n", "\n")
    else:
        llm = load_model(model)
        response = llm([prompt])[0]
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("OUTPUT:", "").strip()
    claims = response.split("\n")
    return claims

def is_claim_relevant(text: str, rating: str, claim: str, model: str = default_model):
    
    prompt = relevance_emotion.format(text, rating, claim)
    if model == default_model:
        response = query_openai(prompt).replace("\n\n", "\n")
    else:
        llm = load_model(model)
        response = llm([prompt])[0]
        
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


def distill_relevant_features(example: EmotionExample, model: str = default_model):
    relevant_claims = []
    for claim in tqdm(example.claims):
        relevance, reasoning = is_claim_relevant(example.text, example.llm_label, claim, model=model)
        if relevance is None:
            continue
        if relevance == "Yes":
            relevant_claims.append(claim)
    return relevant_claims


def calculate_expert_alignment_score(claim: str, model: str = default_model):
    """
    Returns: (category: Optional[str], alignment_score: {'none','partial','complete'}, reasoning: str)
    Robust parsing that tolerates extra text, variant keys, and JSON-like outputs.
    """
    # Build + query
    prompt = alignment_emotion.format(claim)
    if model == default_model:
        response = query_openai(prompt)
    else:
        llm = load_model(model)
        response = llm([prompt])[0]

    if not response or response == "ERROR":
        print("Error in querying OpenAI API")
        return None, "none", ""

    text = str(response).strip()

    # -------------------------------------------------------
    # 1) Try JSON(-ish) first (including code fences)
    # -------------------------------------------------------
    def _maybe_parse_json(s: str):
        # Strip common code fences
        s = s.strip()
        fence = "```"
        if s.startswith(fence) and s.endswith(fence):
            s = s.strip("`")
        # Try to locate a JSON object/array substring
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        # Try raw full string as JSON
        try:
            return json.loads(s)
        except Exception:
            return None

    parsed_json = _maybe_parse_json(text)

    # Accept a few reasonable label variants
    KEY_ALIASES = {
        "category": {"category"},
        "alignment": {"alignment", "category alignment", "category alignment rating", "alignment rating"},
        "reasoning": {"reasoning", "rationale", "explanation"},
    }

    # Normalize to one of allowed alignments
    def _norm_align(a: str) -> str:
        if not a:
            return "none"
        a = a.strip().lower()
        if "complete" in a:
            return "complete"
        if "partial" in a:
            return "partial"
        if "none" in a:
            return "none"
        # map a few common variants
        if a in {"full", "fully", "yes", "aligned", "high"}:
            return "complete"
        if a in {"some", "part", "medium"}:
            return "partial"
        if a in {"no", "low", "not aligned"}:
            return "none"
        return "none"

    def _which_field(k: str) -> str | None:
        k_norm = k.strip().lower()
        for field, aliases in KEY_ALIASES.items():
            if k_norm in aliases:
                return field
        return None

    category = None
    alignment_score = "none"
    reasoning = ""

    # -------------------------------------------------------
    # 2) If we got JSON, try to read fields from it
    # -------------------------------------------------------
    if isinstance(parsed_json, dict):
        # Flatten one level of dict if the model wrapped the result
        cand = parsed_json
        # Find potential keys by alias
        for k, v in list(cand.items()):
            field = _which_field(str(k))
            if field == "category":
                category = str(v).strip()
            elif field == "alignment":
                alignment_score = _norm_align(str(v))
            elif field == "reasoning":
                reasoning = str(v).strip()

    elif isinstance(parsed_json, list) and parsed_json and isinstance(parsed_json[0], dict):
        # Take the first dict-like item
        cand = parsed_json[0]
        for k, v in list(cand.items()):
            field = _which_field(str(k))
            if field == "category":
                category = str(v).strip()
            elif field == "alignment":
                alignment_score = _norm_align(str(v))
            elif field == "reasoning":
                reasoning = str(v).strip()

    # -------------------------------------------------------
    # 3) If still missing, parse "Key: Value" / "Key - Value" lines
    # -------------------------------------------------------
    if category is None or reasoning == "" or alignment_score == "none":
        # Split into non-empty lines and normalize
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Regex to split "Key: Value" OR "Key - Value" (first separator only)
        kv_re = re.compile(r"^\s*([^:\-]+)\s*[:\-]\s*(.+?)\s*$")

        for ln in lines:
            m = kv_re.match(ln)
            if not m:
                continue
            key, value = m.group(1), m.group(2)
            field = _which_field(key)
            if field is None:
                continue

            if field == "category" and not category:
                category = value.strip()
            elif field == "alignment":
                alignment_score = _norm_align(value)
            elif field == "reasoning" and not reasoning:
                reasoning = value.strip()

    # -------------------------------------------------------
    # 4) Final cleanup + fuzzy category mapping
    # -------------------------------------------------------
    # Replace odd hyphen
    if category:
        category = category.replace("-", "-").strip()

    # Fuzzy map to your canonical list (keep your existing logic)
    if category:
        try:
            for c in categories_list:
                if fuzz.ratio(c.lower(), category.lower()) > 90:
                    category = c
                    break
            if category not in categories_list:
                category = None
        except Exception:
            category = None

    # Reasoning sanity: ensure it's not trivially short
    if not reasoning or len(reasoning.strip()) < 5:
        # Try to salvage something sensible (e.g., the first non-key line)
        # Fallback to empty string if nothing reasonable
        reasoning = reasoning.strip()

    # Ensure alignment is one of the allowed strings
    if alignment_score not in {"none", "partial", "complete"}:
        alignment_score = _norm_align(alignment_score)

    # Safety guard (match your original try/except intent, but non-fatal)
    if alignment_score not in {"none", "partial", "complete"}:
        alignment_score = "none"

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


def run_pipeline(emotion_data, baseline="vanilla", model="gpt-4o", evaluation_model: str = default_model):
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
        claims = isolate_individual_features(example.llm_explanation, model=evaluation_model)
        if claims is None:
            continue
        example.claims = [claim.strip() for claim in claims]

    for example in emotion_examples:
        relevant_claims = distill_relevant_features(example, model=evaluation_model)
        example.relevant_claims = relevant_claims

    for example in emotion_examples:
        alignment_scores = []
        alignment_categories = []
        alignment_reasonings = []
        for claim in tqdm(example.relevant_claims):
            category, alignment_score, reasoning = calculate_expert_alignment_score(claim, model=evaluation_model)
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


def aggregate_alignment_scores(alignment_scores, total_claims):
    score_map = {
        "none": 0.0,
        "partial": 0.5,
        "complete": 1.0
    }
    if total_claims == 0:
        return 0.0
    total_score = sum([score_map[score] for score in alignment_scores])
    return total_score / total_claims

def recalculate_alignment(emotion_data, baseline="vanilla", model="gpt-4o"):
    results_dict = {}
    with open("../results/{}/emotion_{}.json".format(baseline, model), 'r') as f:
        results_dict = json.load(f)
    
    emotion_examples = []
    for res in results_dict:
        example = EmotionExample(
            text=res['text'],
            ground_truth=res['ground_truth'],
            llm_label=res['llm_label'],
            llm_explanation=res['llm_explanation']
        )
        example.accuracy = res['accuracy']
        example.claims = res['claims']
        example.relevant_claims = res['relevant_claims']
        emotion_examples.append(example)
    
    print("Recalculating alignment scores for {} examples: {}, {}".format(len(emotion_examples), baseline, model))

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
        example.alignment_reasonings = alignment_reasonings
        example.final_alignment_score = aggregate_alignment_scores(alignment_scores, len(example.claims))

    data_to_save = [example.to_dict() for example in emotion_examples]
    with open("../results/{}/emotion_{}.json".format(baseline, model), 'w') as f:
        json.dump(data_to_save, f, indent=4)


if __name__ == "__main__":
    emotion_data = load_emotion_data()

    #model = "gemini-2.0-flash"
    recalculate_alignment(emotion_data, baseline="vanilla", model="gemini-2.0-flash")
    recalculate_alignment(emotion_data, baseline="cot", model="gemini-2.0-flash")
    recalculate_alignment(emotion_data, baseline="socratic", model="gemini-2.0-flash")
    recalculate_alignment(emotion_data, baseline="subq", model="gemini-2.0-flash")

    # #model = "o1"
    recalculate_alignment(emotion_data, baseline="vanilla", model="o1")
    recalculate_alignment(emotion_data, baseline="cot", model="o1")
    recalculate_alignment(emotion_data, baseline="socratic", model="o1")
    recalculate_alignment(emotion_data, baseline="subq", model="o1")

    # #model = "claude-3-5-sonnet-latest"
    recalculate_alignment(emotion_data, baseline="vanilla", model="claude-3-5-sonnet-latest")
    recalculate_alignment(emotion_data, baseline="cot", model="claude-3-5-sonnet-latest")
    recalculate_alignment(emotion_data, baseline="socratic", model="claude-3-5-sonnet-latest")
    recalculate_alignment(emotion_data, baseline="subq", model="claude-3-5-sonnet-latest")

    # #model = "gpt-4o"
    recalculate_alignment(emotion_data, baseline="vanilla", model="gpt-4o")
    recalculate_alignment(emotion_data, baseline="cot", model="gpt-4o")
    recalculate_alignment(emotion_data, baseline="socratic", model="gpt-4o")
    recalculate_alignment(emotion_data, baseline="subq", model="gpt-4o")
