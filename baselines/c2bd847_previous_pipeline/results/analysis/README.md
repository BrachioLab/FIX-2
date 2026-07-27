# Ablation and Analysis Notes

## Massmaps Legacy Scores by Joint-Claim-Need Label

This analysis groups legacy c2bd847 massmaps evaluation scores by the four joint-claim-need labels:

- `need`
- `not_need`
- `borderline`
- `no_claims`

The labels are read from:

```text
baselines/c2bd847_previous_pipeline/results/analysis/massmaps_joint_claim_need/massmaps_joint_claim_need_details.json
```

The legacy c2bd847 evaluation outputs are read from:

```text
baselines/c2bd847_previous_pipeline/notebooks/_dump/massmaps/final/legacy-c2bd847/gpt-5-mini-2025-08-07/vanilla/eval.gpt-5-mini-2025-08-07/
```

### Step 1: Run Legacy Evaluation

This reuses the saved vanilla explanations from:

```text
results/vanilla/massmaps_gpt-5-mini-2025-08-07.json
```

It skips any per-example JSON files that already exist.

```bash
PYTHON_BIN=/home/weiqiuy/miniconda3/envs/py310/bin/python \
scripts/massmaps/legacy/massmaps_legacy_eval_gpt-5-mini-2025-08-07_vanilla_eval-gpt-5-mini-2025-08-07.sh
```

### Step 2: Summarize by Label

```bash
PYTHON_BIN=/home/weiqiuy/miniconda3/envs/py310/bin/python \
scripts/massmaps/legacy/massmaps_summarize_legacy_by_joint_claim_need_gpt-5-mini-2025-08-07_vanilla.sh
```

The summary script reports two score views:

- `legacy_example_final_alignment_score`: the legacy example-level final score, grouped by each example-category pair's joint-claim-need label.
- `legacy_matching_category_claim_score`: the mean legacy claim-level alignment score for claims whose legacy alignment category matches the current criterion/category for that pair.

Outputs are saved under:

```text
baselines/c2bd847_previous_pipeline/results/analysis/massmaps_legacy_by_joint_claim_need/
```

Files:

```text
massmaps_legacy_by_joint_claim_need_summary.json
massmaps_legacy_by_joint_claim_need_details.json
massmaps_legacy_by_joint_claim_need_details.csv
```
