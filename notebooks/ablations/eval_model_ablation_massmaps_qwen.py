# Uncomment line below to install exlib
# !pip install diskcache
import sys; 

ROOT_DIR = '../..'
sys.path.append(f'{ROOT_DIR}/src')



import openai
import os
import json

def load_api_keys(root_dir):
    import json
    with open(f"{root_dir}/API_KEYS2.json", "r") as file:
        api_keys = json.load(file)
    os.environ['OPENAI_API_KEY'] = api_keys['OPENAI_API_KEY']
    os.environ['ANTHROPIC_API_KEY'] = api_keys['ANTHROPIC_API_KEY']
    os.environ['LLMS_CACHE_PATH'] = "massmaps_qwen.cache"
    # os.environ['GOOGLE_API_KEY'] = api_keys['GOOGLE_API_KEY']
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(root_dir, api_keys['GOOGLE_APPLICATION_CREDENTIALS'])
    os.environ['CACHE_DIR'] = os.path.join(root_dir, 'cache_dir3')
    return api_keys

load_api_keys(ROOT_DIR);

import torch
from datasets import load_dataset

test_dataset = load_dataset("BrachioLab/massmaps-cosmogrid-100k", split='test')
test_dataset.set_format('torch', columns=['input', 'label'])

# import importlib
import sys; sys.path.append("../src")
# import massmaps
# importlib.reload(massmaps)
from massmaps import MassMapsExample
from massmaps import massmap_to_pil_norm, get_llm_generated_answer, get_llm_output
from massmaps import isolate_individual_features, distill_relevant_features, calculate_expert_alignment_score, group_claims_by_category, make_alignment_matrix, categories_list
from llms import load_model

from tqdm.auto import tqdm
import json

# model = 'gpt-4o'
models = [
    "gpt-5.2-pro-2025-12-11",
    "gpt-5-mini-2025-08-07",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-pro",
    # "gemini-2.5-flash"
]

eval_model = load_model("Qwen/Qwen2.5-VL-7B-Instruct", max_model_len=16384)
eval_model_name = 'qwen2.5-vl'

methods = [
    'vanilla', 
    # 'cot', 
    # 'socratic', 
    # 'subq'
]

import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import json
import copy
from tqdm.auto import tqdm
import numpy as np

for model in models:
    print(f"=== Using model {model} ===")
    for method in methods:
        print(f"=== Using method {method} ===")

        load_path = os.path.join(ROOT_DIR, f'results/{method}/massmaps_{model}.json')
        save_path = os.path.join(ROOT_DIR, f'results/{method}/massmaps_{model}_{eval_model_name}.json')

        with open(load_path) as input_file:
            results = json.load(input_file)

        new_results = []

        num_examples = len(results)
        for di in tqdm(range(num_examples)):
            result = results[di]
            
            example_dict = result
            
            example = MassMapsExample(
                input = torch.tensor(example_dict['input']).to(device),
                answer = example_dict['answer'],
                llm_answer = example_dict['llm_answer'],
                llm_explanation = example_dict['llm_explanation'],
            )
            
            # isolate individual features
            claims = isolate_individual_features(example.llm_explanation, model=eval_model)
            if claims is None:
                continue
            example.claims = [claim.strip() for claim in claims]

            # distill relevant features
            relevant_claims = distill_relevant_features(
                example.input, 
                example.llm_answer,
                example.claims,
                model=eval_model
            )
            example.relevant_claims = relevant_claims

            # calculate expert alignment scores
#             align_infos = calculate_expert_alignment_scores(
#                 example.relevant_claims, 
#                 eval_model,
#             )

#             alignable_claims = [info["Claim"] for info in align_infos]
#             alignment_categories = [info["Category"] for info in align_infos]
#             aligned_category_ids = [info["Category ID"] for info in align_infos]
#             alignment_scores = [info["Alignment"] for info in align_infos]
#             alignment_raws = [info["Alignment Raw"] for info in align_infos]
#             alignment_reasonings = [info["Reasoning"] for info in align_infos]
            
#             example.alignable_claims = alignable_claims
#             example.alignment_categories = alignment_categories
#             example.aligned_category_ids = aligned_category_ids
#             example.alignment_scores = alignment_scores
#             example.alignment_raws = alignment_raws
#             example.alignment_reasonings = alignment_reasonings
# calculate expert alignment scores
            claims_by_category, category_alignment_scores, category_alignment_reasonings = calculate_expert_alignment_score(
                relevant_claims, 
                model=eval_model,
                # verbose=True
            )

            example.claims_by_category = claims_by_category
            example.category_alignment_scores = category_alignment_scores
            example.category_alignment_reasonings = category_alignment_reasonings

            alignment_matrix = make_alignment_matrix(
                example.claims,
                claims_by_category,
                category_alignment_scores
            )

            final_alignment_score = alignment_matrix.max(axis=-1).mean()
            if np.isnan(final_alignment_score):
                print(f'example {idx} final_alignment_score is NaN')
            example.final_alignment_score = final_alignment_score
            
            # save
            save_dict = {}
            for k, v in example.__dict__.items():
                save_dict[k] = v if not isinstance(v, torch.Tensor) else v.cpu().numpy().tolist()
            # with open(save_path, 'wt') as output_file:
            #     json.dump(save_dict, output_file)

            new_results.append(save_dict)


        with open(save_path, 'wt') as output_file:
            json.dump(new_results, output_file, indent=4)