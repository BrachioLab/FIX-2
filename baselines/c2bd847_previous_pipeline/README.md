# Frozen c2bd847 Previous Pipeline

This folder preserves the pipeline code and notebooks from commit `c2bd84725f49939be34b75b44d94ea4822e07b0c` under a separate baseline name.

The copied prompts live in `legacy_src/prompts/` and are intentionally separate from the current repo prompts. `run_baseline.py` puts `legacy_src` first on `sys.path`, so baseline runs use the frozen prompt text.

## Usage

List the current default model names:

```bash
python baselines/c2bd847_previous_pipeline/run_baseline.py --dataset massmaps --list-models
```

Run one dataset with one model:

```bash
python baselines/c2bd847_previous_pipeline/run_baseline.py \
  --dataset massmaps \
  --method vanilla \
  --model gpt-5-mini-2025-08-07 \
  --eval-model gpt-5-mini-2025-08-07 \
  --num-samples 3
```

Valid datasets are `massmaps`, `cholec`, `cardiac`, `sepsis`, `supernova`, `emotion`, and `politeness`.

Valid methods are `vanilla`, `cot`, `socratic`, and `subq`.

The runner accepts any model supported by `legacy_src/llms.py`, including the current defaults:

```text
gpt-5.2-pro-2025-12-11
gpt-5-mini-2025-08-07
claude-opus-4-5-20251101
claude-haiku-4-5-20251001
gemini-2.5-pro
gemini-2.5-flash
```

Outputs are written under `baselines/c2bd847_previous_pipeline/results/`.

## Notes

- The old notebooks were copied into `notebooks/`.
- The old prompts were not edited.
- `legacy_src/llms.py` is adapted from the current repo so newer model names and provider routing work with the previous pipeline.
