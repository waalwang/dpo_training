# DPO Training — Trajectory-Level Preference Learning

DPO/IPO fine-tuning on multi-turn conversation trajectories extracted from Reddit and Hacker News comment trees. Supports QLoRA and full fine-tuning across multiple model sizes, and an optional two-stage SFT -> DPO pipeline.

## Project Structure

```
dpo_training/
├── configs/
│   ├── default.yaml          # Training config, model presets, hardware profiles
│   └── deepspeed_z3.yaml     # DeepSpeed ZeRO-3 config for full fine-tune
├── data_loader.py             # Loads trajectory parquet → HF Dataset for DPOTrainer
├── train.py                   # Main training script (QLoRA / full FT)
├── scripts/
│   ├── eval.py                # Evaluation (implicit reward, RM scoring, best-of-N)
│   └── inspect_data.py        # Dataset sanity check and stats
├── requirements.txt
└── data/                      # (created at runtime)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# For full fine-tune on multi-GPU:
pip install deepspeed
```

## Data

Trajectory parquet files are produced by the companion crawler project (`conversation_crawler`). Default path: `../crawler/conversation_crawler/data/trajectory/`.

Each row contains:
- `prompt`: shared conversation prefix (list of `{role, content}` turns)
- `chosen`: prefix + high-quality trajectory branch
- `rejected`: prefix + low-quality trajectory branch

Inspect data before training:
```bash
python scripts/inspect_data.py
python scripts/inspect_data.py --source hacker_news --show-samples 3
```

## Hardware Profiles

| Profile | Mode | Model Size | Hardware | Notes |
|---------|------|-----------|----------|-------|
| `local` | QLoRA | 1.5B | GTX 1080Ti 11GB | fp16, eager attn, 2048 ctx |
| `cloud` | QLoRA | 1.5B–7B | 1xA100 40GB | bf16, FA2, 8192 ctx |
| `cloud_qlora_big` | QLoRA | 14B | 1xA100 80GB | bf16, FA2, 8192 ctx |
| `cloud_full` | Full FT | 7B–8B | 2xA100 80GB | bf16, FA2, DeepSpeed ZeRO-3 |

## Model Presets

Use `--model-override <key>` to swap the backbone:

| Key | Model |
|-----|-------|
| `qwen-1.5b` | Qwen/Qwen2.5-1.5B-Instruct (default) |
| `qwen-7b` | Qwen/Qwen2.5-7B-Instruct |
| `qwen-14b` | Qwen/Qwen2.5-14B-Instruct |
| `llama-8b` | NousResearch/Meta-Llama-3.1-8B-Instruct or meta-llama/Llama-3.1-8B-Instruct |

## Training

### Stages

| `--stage` | What runs |
|-----------|-----------|
| `dpo` (default) | DPO/IPO only, starting from the base model |
| `sft` | SFT on chosen trajectories only, saves to `outputs/sft/final` |
| `both` | SFT then DPO in one run, DPO initialised from the SFT checkpoint |

### Local — QLoRA on Qwen2.5-1.5B (GTX 1080Ti 11GB)

```bash
# Dry run (verify data + model loading):
python train.py --profile local --dry-run

# DPO only:
python train.py --profile local

# Two-stage SFT -> DPO:
python train.py --profile local --stage both
```

### Cloud — QLoRA on Qwen2.5-1.5B (1xA100 40GB)

```bash
# DPO only:
python train.py --profile cloud

# Two-stage SFT -> DPO:
python train.py --profile cloud --stage both

# Run stages separately (e.g. to tune each independently):
python train.py --profile cloud --stage sft
python train.py --profile cloud --stage dpo --sft-checkpoint outputs/sft/final
```

### Cloud — QLoRA on Qwen2.5-14B (1xA100 80GB)

```bash
python train.py --profile cloud_qlora_big --model-override qwen-14b
python train.py --profile cloud_qlora_big --model-override qwen-14b --stage both
```

### Cloud — Full fine-tune on Qwen2.5-7B (2xA100 80GB)

```bash
accelerate launch --config_file configs/deepspeed_z3.yaml \
    train.py --profile cloud_full --model-override qwen-7b
```

### Cloud — Full fine-tune on LLaMA 3.1-8B (2xA100 80GB)

```bash
accelerate launch --config_file configs/deepspeed_z3.yaml \
    train.py --profile cloud_full --model-override llama-8b
```

## Evaluation

`scripts/eval.py` runs the same metric suite on either a DPO checkpoint or the raw pretrained baseline. Pass exactly one of `--checkpoint` or `--baseline`. Running both modes on the same profile and reward model gives directly comparable numbers, so the DPO gain is just `dpo - baseline`.

### Basic - implicit reward accuracy + generation samples

```bash
# DPO checkpoint:
python scripts/eval.py --checkpoint outputs/final --profile local

# Pretrained baseline (default Qwen2.5-1.5B-Instruct):
python scripts/eval.py --baseline --profile local
```

### Pretrained baselines for each planned backbone

Use `--model-override` with a key from the `model_presets` block in `configs/default.yaml` to swap backbones. The baseline loads with the same 4-bit quantization as the QLoRA path (or bf16 under `cloud_full`), so baseline numerics match the DPO run on the same profile.

```bash
python scripts/eval.py --baseline --profile local                                    # qwen-1.5b (default)
python scripts/eval.py --baseline --profile cloud --model-override qwen-7b
python scripts/eval.py --baseline --profile cloud_qlora_big --model-override qwen-14b
python scripts/eval.py --baseline --profile cloud_full --model-override llama-8b
```

Baseline results are written to `outputs/baseline_<preset>/eval_results.json` and do not collide with DPO checkpoint outputs.

### With reward model scoring

Score outputs against your trained reward model. Accepts a local path or HuggingFace Hub ID. Works with both `--checkpoint` and `--baseline`:

```bash
python scripts/eval.py --checkpoint outputs/final --profile local \
    --reward-model /path/to/reward_model_checkpoint

python scripts/eval.py --baseline --profile local \
    --reward-model /path/to/reward_model_checkpoint
```

This adds:
- **RM accuracy on held-out pairs** - does the reward model agree with ground-truth preferences?
- **RM-scored generations** - per-sample reward scores for model outputs vs reference chosen

### Extra metrics: diversity, novelty, naturalness, tone

`--extra-metrics` runs the full auxiliary suite on whichever model is under eval (baseline backbone or DPO checkpoint), so a pre-training backbone scan produces numbers directly comparable to the post-training DPO run:

- **Distinct-1, Distinct-2** - unique n-gram ratio across generations.
- **Self-BLEU** - intra-set repetition across k samples per prompt (requires `sacrebleu`; skipped gracefully if missing).
- **MAUVE** - distribution gap vs held-out human chosen continuations (requires `mauve-text`; skipped gracefully if missing).
- **Training-set n-gram overlap** - memorization proxy: fraction of generated 8-grams found in the training corpus.
- **Formality classifier** - mean probability that outputs are informal (casual-tone proxy, uses `s-nlp/roberta-base-formality-ranker`).

Install optional deps (uncomment in `requirements.txt` or install directly):

```bash
pip install sacrebleu mauve-text
```

Backbone scan before training, one per planned model:

```bash
python scripts/eval.py --baseline --profile local --extra-metrics
python scripts/eval.py --baseline --profile cloud --model-override qwen-7b --extra-metrics
python scripts/eval.py --baseline --profile cloud_qlora_big --model-override qwen-14b --extra-metrics
python scripts/eval.py --baseline --profile cloud_full --model-override llama-8b --extra-metrics
```

Same flag on a DPO checkpoint for the post-training comparison:

```bash
python scripts/eval.py --checkpoint outputs/final --profile local --extra-metrics
```

Knobs:

| Flag | Default | Description |
|------|---------|-------------|
| `--extra-metrics` | False | Enable the full extra-metrics suite |
| `--diversity-k` | 8 | Samples per prompt for diversity |
| `--diversity-prompts` | 30 | Number of test prompts used for diversity/MAUVE |
| `--novelty-ngram` | 8 | N-gram size for training-set overlap |
| `--novelty-train-docs` | 5000 | Max training docs used to build the n-gram set |

### Best-of-N generation

Generate N candidates per prompt, pick the best one by reward model score:

```bash
python scripts/eval.py --checkpoint outputs/final --profile local \
    --reward-model /path/to/reward_model_checkpoint --best-of-n 8
```

### Eval flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | None | DPO model path. Mutually exclusive with `--baseline`. |
| `--baseline` | False | Evaluate the raw pretrained model from `cfg['model']['name']`. Mutually exclusive with `--checkpoint`. |
| `--model-override` | None | Swap the base model via a `model_presets` key (`qwen-1.5b`, `qwen-7b`, `qwen-14b`, `llama-8b`). |
| `--profile` | `local` | Hardware profile |
| `--reward-model` | None | Reward model path (local or HF Hub) |
| `--best-of-n` | 0 | N candidates for best-of-N (requires `--reward-model`) |
| `--num-samples` | 10 | Number of generation samples |
| `--max-eval` | 500 | Max examples for accuracy computation |
| `--output` | auto | Path to save results JSON |
| `--extra-metrics` | False | Run MAUVE + diversity + training overlap + formality |
| `--diversity-k` | 8 | Samples per prompt for diversity metrics |
| `--diversity-prompts` | 30 | Number of test prompts for diversity/MAUVE |
| `--novelty-ngram` | 8 | N-gram size for training-set overlap |
| `--novelty-train-docs` | 5000 | Max training docs used for the n-gram set |

## Vast.ai Quick Start

1. Rent an instance (e.g. 1xA100 80GB for QLoRA 14B, 2xA100 80GB for full FT 7B)
2. Upload this project + trajectory data
3. Install deps: `pip install -r requirements.txt && pip install deepspeed`
4. Run training with the appropriate profile
5. Download the checkpoint: `scp -r <instance>:dpo_training/outputs/final ./outputs/`
6. Evaluate locally: `python scripts/eval.py --checkpoint outputs/final --profile local`
