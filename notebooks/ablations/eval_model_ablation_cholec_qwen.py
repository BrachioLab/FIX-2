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

import importlib
import sys; sys.path.append("../src")
import cholec
importlib.reload(cholec)
from cholec import get_llm_generated_answer
from cholec import CholecExample, CholecDataset, load_model #, items_to_examples
from cholec import isolate_individual_features, distill_relevant_features, calculate_expert_alignment_score, group_claims_by_category, make_alignment_matrix, categories_list

test_dataset = CholecDataset(split="test")

from tqdm.auto import tqdm
import json

# model = 'gpt-4o'
models = [
    "gpt-5.2-pro-2025-12-11",
    "gpt-5-mini-2025-08-07",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-pro",
    "gemini-2.5-flash"
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

# for model in models:
#     print(f"=== Using model {model} ===")
#     for method in methods:
        # print(f"=== Using method {method} ===")

model = models[0]
method = methods[0]
    
load_path = os.path.join(ROOT_DIR, f'results/{method}/cholec_{model}.json')
save_path = os.path.join(ROOT_DIR, f'results/{method}/cholec_{model}_{eval_model_name}.json')

with open(load_path) as input_file:
    results = json.load(input_file)
    
results[0].keys()

id2idx_mapping = {
    test_dataset.dataset[i]['id']: i
    for i in range(len(test_dataset.dataset))
}

import json
import copy
import numpy as np
from tqdm.auto import tqdm

for model in models:
    print(f"=== Using model {model} ===")
    for method in methods:
        print(f"=== Using method {method} ===")

        load_path = os.path.join(ROOT_DIR, f'results/{method}/cholec_{model}.json')
        save_path = os.path.join(ROOT_DIR, f'results/{method}/cholec_{model}_{eval_model_name}.json')

        with open(load_path) as input_file:
            results = json.load(input_file)

        new_results = []

        num_examples = len(results)
        for di in tqdm(range(num_examples)):
            result = results[di]
            
            image = test_dataset[id2idx_mapping[result['id']]]['image']
            
            example = CholecExample(
                id=result['id'],
                image = image,
                true_safe_list=result['true_safe_list'],
                true_unsafe_list=result['true_unsafe_list'],
                llm_raw_output=result['llm_raw_output'],
                llm_explanation=result['llm_explanation'],
                llm_safe_list=result['llm_safe_list'],
                llm_unsafe_list=result['llm_unsafe_list'],
            )
            example.safe_iou = result['safe_iou']
            example.unsafe_iou = result['unsafe_iou']
            
            
            # isolate individual features
            all_claims = isolate_individual_features(example.llm_explanation, model=eval_model)
            if all_claims is None:
                continue
            example.all_claims = [claim.strip() for claim in all_claims]

            # distill relevant features
            relevant_claims = distill_relevant_features(
                example.image, 
                example.all_claims,
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
            
#             # Non-alignable claims are given a score of 0.0
#             if len(align_infos) > 0:
#                 example.final_alignment_score = sum(info["Alignment"] for info in align_infos) / len(example.all_claims)
#             else:
#                 example.final_alignment_score = 0.0
    
            claims_by_category, category_alignment_scores, category_alignment_reasonings = calculate_expert_alignment_score(
                relevant_claims, 
                model=eval_model,
                # verbose=True
            )

            example.claims_by_category = claims_by_category
            example.category_alignment_scores = category_alignment_scores
            example.category_alignment_reasonings = category_alignment_reasonings

            alignment_matrix = make_alignment_matrix(
                example.all_claims,
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
                if not isinstance(v, torch.Tensor):
                    save_dict[k] = v # if not isinstance(v, torch.Tensor) else v.cpu().numpy().tolist()
            # with open(save_path, 'wt') as output_file:
            #     json.dump(save_dict, output_file)

            new_results.append(save_dict)


        with open(save_path, 'wt') as output_file:
            json.dump(new_results, output_file, indent=4)