"""
scripts/eval.py

Evaluate a DPO-trained model, or the raw pretrained baseline:
  1. Implicit reward accuracy -- how often the model assigns higher
     log-prob to chosen vs rejected (no external model needed)
  2. Reward model scoring -- use your trained reward model to score
     generated outputs vs reference chosen/rejected
  3. Best-of-N generation -- generate N candidates, pick the best one
     according to the reward model
  4. Generation samples -- qualitative review of continuations

Pass exactly one of --checkpoint or --baseline. --baseline runs the
identical metric suite on the pretrained model from cfg["model"]["name"]
(optionally swapped via --model-override), so baseline and DPO numbers
are directly comparable.

Usage:
    # DPO checkpoint eval:
    python scripts/eval.py --checkpoint outputs/final --profile local

    # Baseline eval on the default pretrained model:
    python scripts/eval.py --baseline --profile local

    # Baseline eval on a specific preset (qwen-1.5b, qwen-7b, qwen-14b, llama-8b):
    python scripts/eval.py --baseline --profile cloud --model-override qwen-7b

    # Any mode with reward model scoring:
    python scripts/eval.py --checkpoint outputs/final --profile local \
        --reward-model /path/to/reward_model_checkpoint

    # Best-of-N (requires --reward-model):
    python scripts/eval.py --baseline --profile local \
        --reward-model /path/to/reward_model_checkpoint --best-of-n 8

    # Reward model can also be an HF Hub ID:
    python scripts/eval.py --checkpoint outputs/final --profile local \
        --reward-model your-username/reward-model-1.5b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
import yaml
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / model loading
# ---------------------------------------------------------------------------

def load_config(config_path: str, profile: str, model_override: str | None = None) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    hw = cfg["profiles"][profile]
    cfg["training"].update(hw)
    cfg["_profile"] = profile
    cfg["_full_finetune"] = hw.get("full_finetune", False)
    if model_override:
        presets = cfg.get("model_presets", {})
        if model_override not in presets:
            raise ValueError(
                f"Unknown --model-override '{model_override}'. "
                f"Valid keys: {list(presets.keys())}"
            )
        cfg["model"].update(presets[model_override])
        cfg["_model_override"] = model_override
    return cfg


def _bnb_config(bf16: bool) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(checkpoint_dir: str | None, cfg: dict):
    """Load a model for eval.

    checkpoint_dir=None -> raw pretrained baseline from cfg["model"]["name"]
    checkpoint_dir set + full_ft -> full fine-tune checkpoint loaded directly
    checkpoint_dir set + QLoRA   -> base model + LoRA adapter from checkpoint

    Baseline uses the same 4-bit quantization as the QLoRA path so that
    baseline and DPO numbers are computed under matched numerical precision.
    """
    model_name = cfg["model"]["name"]
    t = cfg["training"]
    full_ft = cfg["_full_finetune"]

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    attn_impl = t.get("attn_implementation")
    if attn_impl and attn_impl != "eager":
        model_kwargs["attn_implementation"] = attn_impl

    if checkpoint_dir is None:
        # Baseline: raw pretrained model, no adapter, no fine-tune
        if full_ft:
            model_kwargs["torch_dtype"] = torch.bfloat16 if t["bf16"] else torch.float16
        else:
            model_kwargs["quantization_config"] = _bnb_config(t["bf16"])
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        tokenizer_src = model_name
    elif full_ft:
        # Full fine-tune: load directly from checkpoint
        model_kwargs["torch_dtype"] = torch.bfloat16 if t["bf16"] else torch.float16
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, **model_kwargs)
        tokenizer_src = checkpoint_dir
    else:
        # QLoRA: load base + adapter
        model_kwargs["quantization_config"] = _bnb_config(t["bf16"])
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        tokenizer_src = model_name

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def load_reward_model(reward_model_path: str, bf16: bool = False):
    """Load a reward model (local path or HF Hub ID).

    Tries AutoModelForSequenceClassification first (standard reward model).
    Falls back to causal LM with a value head (TRL-trained).
    """
    logger.info(f"Loading reward model from: {reward_model_path}")

    bnb = _bnb_config(bf16)
    model_kwargs = {
        "quantization_config": bnb,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path, **model_kwargs
        )
        model_type = "sequence_classification"
    except (ValueError, OSError):
        # Fallback: TRL's AutoModelForCausalLMWithValueHead
        from trl import AutoModelForCausalLMWithValueHead
        model_kwargs.pop("quantization_config")
        model_kwargs["torch_dtype"] = torch.bfloat16 if bf16 else torch.float16
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            reward_model_path, **model_kwargs
        )
        model_type = "value_head"

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(reward_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info(f"Reward model loaded (type: {model_type})")
    return model, tokenizer, model_type


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sequence_logprob(model, tokenizer, messages: list[dict]) -> float:
    """Mean log-probability of a message sequence under a causal LM."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]
    labels = inputs["input_ids"][:, 1:]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    mask = labels != tokenizer.pad_token_id
    mean_lp = (token_log_probs * mask).sum() / mask.sum()
    return mean_lp.item()


@torch.no_grad()
def _reward_score(model, tokenizer, model_type: str, messages: list[dict]) -> float:
    """Score a conversation using the reward model."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if model_type == "sequence_classification":
        outputs = model(**inputs)
        return outputs.logits[:, -1].item() if outputs.logits.dim() > 1 else outputs.logits.item()
    else:
        # value_head: TRL's model returns (lm_logits, loss, value)
        _, _, values = model(**inputs)
        # Last non-padding token's value
        seq_len = inputs["attention_mask"].sum(dim=1) - 1
        return values[0, seq_len].item()


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_implicit_reward_accuracy(
    model, tokenizer, dataset, max_samples: int = 500
) -> float:
    """Fraction of examples where DPO model prefers chosen over rejected."""
    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        if i >= max_samples:
            break

        chosen_score = _sequence_logprob(model, tokenizer, example["chosen"])
        rejected_score = _sequence_logprob(model, tokenizer, example["rejected"])

        if chosen_score > rejected_score:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  Implicit reward: {i + 1}/{min(len(dataset), max_samples)}")

    return correct / total if total > 0 else 0.0


def _first_turn_pair(example: dict) -> tuple[list[dict], list[dict]] | None:
    """Extract first-turn chosen/rejected from a trajectory example.

    Returns (prompt + first chosen assistant turn, prompt + first rejected assistant turn),
    or None if either trajectory has no assistant turn.
    """
    prompt = example["prompt"]
    prompt_len = len(prompt)

    chosen_traj = example["chosen"][prompt_len:]
    rejected_traj = example["rejected"][prompt_len:]

    # Find first assistant turn in each trajectory
    chosen_first = next((t for t in chosen_traj if t["role"] == "assistant"), None)
    rejected_first = next((t for t in rejected_traj if t["role"] == "assistant"), None)

    if chosen_first is None or rejected_first is None:
        return None

    return (
        prompt + [chosen_first],
        prompt + [rejected_first],
    )


@torch.no_grad()
def compute_reward_model_accuracy(
    rm_model, rm_tokenizer, rm_type: str, dataset, max_samples: int = 500
) -> dict:
    """Use the external reward model to score chosen vs rejected pairs.

    Scores full trajectories. Returns accuracy and mean score delta.
    """
    correct = 0
    total = 0
    deltas = []

    for i, example in enumerate(dataset):
        if i >= max_samples:
            break

        chosen_score = _reward_score(rm_model, rm_tokenizer, rm_type, example["chosen"])
        rejected_score = _reward_score(rm_model, rm_tokenizer, rm_type, example["rejected"])

        delta = chosen_score - rejected_score
        deltas.append(delta)
        if delta > 0:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  RM full-traj: {i + 1}/{min(len(dataset), max_samples)}")

    accuracy = correct / total if total > 0 else 0.0
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return {"accuracy": accuracy, "mean_score_delta": mean_delta, "num_eval": total}


@torch.no_grad()
def compute_first_turn_rm_accuracy(
    rm_model, rm_tokenizer, rm_type: str, dataset, max_samples: int = 500
) -> dict:
    """RM accuracy using only the first assistant turn of each trajectory.

    This is the most in-distribution evaluation for a single-turn reward model,
    since it scores (prompt + one response) pairs -- exactly what the RM was
    trained on.
    """
    correct = 0
    total = 0
    skipped = 0
    deltas = []

    for i, example in enumerate(dataset):
        if i >= max_samples:
            break

        pair = _first_turn_pair(example)
        if pair is None:
            skipped += 1
            continue

        chosen_first, rejected_first = pair
        chosen_score = _reward_score(rm_model, rm_tokenizer, rm_type, chosen_first)
        rejected_score = _reward_score(rm_model, rm_tokenizer, rm_type, rejected_first)

        delta = chosen_score - rejected_score
        deltas.append(delta)
        if delta > 0:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  RM first-turn: {i + 1}/{min(len(dataset), max_samples)}")

    if skipped:
        logger.info(f"  Skipped {skipped} examples with no assistant turn in trajectory")

    accuracy = correct / total if total > 0 else 0.0
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return {"accuracy": accuracy, "mean_score_delta": mean_delta, "num_eval": total}


@torch.no_grad()
def score_generations(
    rm_model, rm_tokenizer, rm_type: str, samples: list[dict]
) -> list[dict]:
    """Score generated outputs with the reward model.

    Adds rm_score_generated, rm_score_chosen, rm_score_delta to each sample.
    """
    for s in samples:
        prompt = s["prompt"]

        # Build full conversations for scoring
        generated_conv = prompt + [{"role": "assistant", "content": s["generated"]}]
        chosen_conv = prompt + [{"role": "assistant", "content": s["chosen_start"]}]

        s["rm_score_generated"] = _reward_score(
            rm_model, rm_tokenizer, rm_type, generated_conv
        )
        s["rm_score_chosen"] = _reward_score(
            rm_model, rm_tokenizer, rm_type, chosen_conv
        )
        s["rm_score_delta"] = s["rm_score_generated"] - s["rm_score_chosen"]

    return samples


@torch.no_grad()
def best_of_n_generation(
    model, tokenizer, rm_model, rm_tokenizer, rm_type: str,
    dataset, n: int = 8, num_samples: int = 10,
) -> list[dict]:
    """Generate N candidates per prompt, pick the best one by reward score."""
    results = []

    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        prompt_messages = example["prompt"]

        text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate N candidates
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
        )

        candidates = []
        for j in range(n):
            gen_text = tokenizer.decode(
                output_ids[j][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            conv = prompt_messages + [{"role": "assistant", "content": gen_text}]
            score = _reward_score(rm_model, rm_tokenizer, rm_type, conv)
            candidates.append({"text": gen_text, "rm_score": score})

        candidates.sort(key=lambda c: c["rm_score"], reverse=True)

        results.append({
            "prompt": prompt_messages,
            "best": candidates[0],
            "worst": candidates[-1],
            "all_scores": [c["rm_score"] for c in candidates],
            "score_spread": candidates[0]["rm_score"] - candidates[-1]["rm_score"],
        })

        logger.info(
            f"  BoN {i + 1}: best={candidates[0]['rm_score']:.3f} "
            f"worst={candidates[-1]['rm_score']:.3f} "
            f"spread={results[-1]['score_spread']:.3f}"
        )

    return results


@torch.no_grad()
def generate_samples(
    model, tokenizer, dataset, num_samples: int = 10
) -> list[dict]:
    """Generate continuations from test prompts."""
    samples = []
    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        prompt_messages = example["prompt"]

        text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        samples.append({
            "prompt": prompt_messages,
            "generated": generated,
            "chosen_start": example["chosen"][len(prompt_messages)]["content"][:200]
            if len(example["chosen"]) > len(prompt_messages) else "",
        })

    return samples


# ---------------------------------------------------------------------------
# Extra metrics: diversity, novelty, naturalness, tone
# ---------------------------------------------------------------------------

def _assistant_text(messages: list[dict]) -> str:
    """Content of the last assistant turn in a message list, or ''."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""


def _tokens(text: str) -> list[str]:
    return text.split()


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


@torch.no_grad()
def generate_multi_samples(
    model, tokenizer, dataset, num_prompts: int, k: int
) -> list[dict]:
    """Generate k samples per prompt for the first num_prompts test examples.

    Returns a list of dicts with the prompt messages and the k generated texts.
    """
    groups = []
    for i in range(min(num_prompts, len(dataset))):
        example = dataset[i]
        prompt_messages = example["prompt"]

        text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {x: v.to(model.device) for x, v in inputs.items()}

        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
        )
        gens = [
            tokenizer.decode(
                output_ids[j][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            for j in range(k)
        ]
        groups.append({"prompt": prompt_messages, "generations": gens})

        if (i + 1) % 10 == 0:
            logger.info(f"  Multi-sample gen: {i + 1}/{num_prompts}")

    return groups


def compute_diversity_metrics(groups: list[dict]) -> dict:
    """Distinct-1, distinct-2 (corpus-level), self-BLEU (per-prompt, averaged).

    Distinct-n: ratio of unique n-grams to total n-grams across all generations.
    Self-BLEU: for each generation in a group, BLEU against the other k-1
        generations of the same prompt, averaged over all (prompt, sample) pairs.
        Higher = more repetitive across samples.
    """
    texts = [t for g in groups for t in g["generations"]]
    token_lists = [_tokens(t) for t in texts]

    unigrams = [tok for toks in token_lists for tok in toks]
    bigrams = [ng for toks in token_lists for ng in _ngrams(toks, 2)]
    distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0.0
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0

    self_bleu = None
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)
        scores = []
        for g in groups:
            gens = g["generations"]
            if len(gens) < 2:
                continue
            for i in range(len(gens)):
                hyp = gens[i]
                refs = [gens[j] for j in range(len(gens)) if j != i]
                scores.append(bleu.sentence_score(hyp, refs).score)
        self_bleu = sum(scores) / len(scores) if scores else None
    except ImportError:
        logger.warning("sacrebleu not installed - skipping self-BLEU (`pip install sacrebleu`)")
    except Exception as e:
        logger.warning(f"self-BLEU computation failed: {e}")

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "self_bleu": self_bleu,
        "num_prompts": len(groups),
        "samples_per_prompt": len(groups[0]["generations"]) if groups else 0,
    }


def compute_training_overlap(
    generations: list[str], training_texts: list[str], n: int = 8,
    max_training_docs: int = 5000,
) -> dict:
    """Fraction of generated n-grams that appear verbatim in the training corpus.

    Memorization proxy: high overlap_rate means the model is reproducing training
    text; low means generations are novel in n-gram space.
    """
    sample = training_texts[:max_training_docs]
    train_ngrams: set = set()
    for t in sample:
        train_ngrams.update(_ngrams(_tokens(t), n))

    total = 0
    overlap = 0
    for g in generations:
        grams = _ngrams(_tokens(g), n)
        total += len(grams)
        overlap += sum(1 for ng in grams if ng in train_ngrams)

    return {
        "ngram_size": n,
        "overlap_rate": overlap / total if total else 0.0,
        "generated_ngrams": total,
        "training_docs_used": len(sample),
        "training_unique_ngrams": len(train_ngrams),
    }


def compute_mauve_score(generations: list[str], references: list[str]) -> dict | None:
    """MAUVE score (Pillutla et al., NeurIPS 2021). Higher = closer to human dist."""
    try:
        import mauve
    except ImportError:
        logger.warning("mauve-text not installed - skipping MAUVE (`pip install mauve-text`)")
        return None
    try:
        out = mauve.compute_mauve(
            p_text=references,
            q_text=generations,
            device_id=0 if torch.cuda.is_available() else -1,
            max_text_length=512,
            verbose=False,
        )
        return {
            "mauve": float(out.mauve),
            "num_generations": len(generations),
            "num_references": len(references),
        }
    except Exception as e:
        logger.warning(f"MAUVE computation failed: {e}")
        return None


def compute_formality_scores(
    texts: list[str],
    model_id: str = "s-nlp/roberta-base-formality-ranker",
) -> dict | None:
    """Mean probability that generations are informal (casual tone proxy)."""
    try:
        from transformers import pipeline
    except ImportError:
        return None
    try:
        clf = pipeline(
            "text-classification",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512,
        )
        preds = clf(texts, batch_size=8)
        informal = []
        for p in preds:
            lbl = p["label"].lower()
            informal.append(p["score"] if "informal" in lbl else 1.0 - p["score"])
        return {
            "mean_informal_prob": sum(informal) / len(informal) if informal else 0.0,
            "num_texts": len(informal),
            "classifier": model_id,
        }
    except Exception as e:
        logger.warning(f"formality classifier failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DPO model or pretrained baseline")
    parser.add_argument("--checkpoint", default=None,
                        help="DPO model path (local or HF Hub). Mutually exclusive with --baseline.")
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate the raw pretrained model from cfg['model']['name'] "
                             "(no adapter, no fine-tune). Mutually exclusive with --checkpoint.")
    parser.add_argument("--model-override", default=None,
                        help="Swap the base model using a model_presets key "
                             "(qwen-1.5b, qwen-7b, qwen-14b, llama-8b).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--profile", default="local",
                        choices=["local", "cloud", "cloud_full", "cloud_qlora_big"])
    parser.add_argument("--reward-model", default=None,
                        help="Reward model path (local or HF Hub). Enables RM scoring.")
    parser.add_argument("--best-of-n", type=int, default=0,
                        help="Generate N candidates and pick best by RM score (requires --reward-model)")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-eval", type=int, default=500)
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    # Extra metrics (diversity, novelty, naturalness, tone). Work identically
    # in --baseline and --checkpoint modes so pre-training backbone scans and
    # post-training DPO evals are directly comparable.
    parser.add_argument("--extra-metrics", action="store_true",
                        help="Run MAUVE, diversity (distinct-n, self-BLEU), "
                             "training-set n-gram overlap, and formality classifier.")
    parser.add_argument("--diversity-k", type=int, default=8,
                        help="Samples per prompt for diversity metrics.")
    parser.add_argument("--diversity-prompts", type=int, default=30,
                        help="Number of test prompts to use for diversity/MAUVE.")
    parser.add_argument("--novelty-ngram", type=int, default=8,
                        help="N-gram size for training-set overlap.")
    parser.add_argument("--novelty-train-docs", type=int, default=5000,
                        help="Max training docs used to build the n-gram set.")
    args = parser.parse_args()

    if args.best_of_n > 0 and not args.reward_model:
        parser.error("--best-of-n requires --reward-model")
    if bool(args.checkpoint) == bool(args.baseline):
        parser.error("Pass exactly one of --checkpoint or --baseline.")

    cfg = load_config(args.config, args.profile, args.model_override)

    # --- Data ---
    logger.info("Loading dataset...")
    dataset = load_from_config(cfg)
    test_set = dataset["test"]
    logger.info(f"Test set: {len(test_set)} examples")

    # --- Model under eval (DPO checkpoint or pretrained baseline) ---
    if args.baseline:
        eval_mode = "baseline"
        logger.info(f"Loading pretrained baseline: {cfg['model']['name']}")
    else:
        eval_mode = "dpo"
        logger.info(f"Loading DPO model from {args.checkpoint}...")
    model, tokenizer = load_model(args.checkpoint, cfg)

    # --- Reward model (optional) ---
    rm_model = rm_tokenizer = rm_type = None
    if args.reward_model:
        bf16 = cfg["training"].get("bf16", False)
        rm_model, rm_tokenizer, rm_type = load_reward_model(args.reward_model, bf16)

    results = {
        "eval_mode": eval_mode,
        "model_name": cfg["model"]["name"],
        "model_override": cfg.get("_model_override"),
        "profile": args.profile,
        "checkpoint": args.checkpoint,
    }

    # --- 1. Implicit reward accuracy ---
    logger.info("Computing implicit reward accuracy (DPO log-prob)...")
    implicit_acc = compute_implicit_reward_accuracy(model, tokenizer, test_set, args.max_eval)
    logger.info(f"Implicit reward accuracy: {implicit_acc:.4f}")
    results["implicit_reward_accuracy"] = implicit_acc

    # --- 2a. RM first-turn accuracy (most reliable -- in-distribution for single-turn RM) ---
    if rm_model:
        logger.info("Computing RM first-turn accuracy (in-distribution)...")
        ft_results = compute_first_turn_rm_accuracy(
            rm_model, rm_tokenizer, rm_type, test_set, args.max_eval
        )
        logger.info(
            f"RM first-turn accuracy: {ft_results['accuracy']:.4f} "
            f"(mean delta: {ft_results['mean_score_delta']:.4f})"
        )
        results["rm_first_turn_accuracy"] = ft_results

    # --- 2b. RM full-trajectory accuracy (directional -- may be out-of-distribution) ---
    if rm_model:
        logger.info("Computing RM full-trajectory accuracy (directional)...")
        rm_results = compute_reward_model_accuracy(
            rm_model, rm_tokenizer, rm_type, test_set, args.max_eval
        )
        logger.info(
            f"RM full-traj accuracy: {rm_results['accuracy']:.4f} "
            f"(mean delta: {rm_results['mean_score_delta']:.4f})"
        )
        results["rm_full_trajectory_accuracy"] = rm_results

    # --- 3. Generation samples ---
    logger.info(f"Generating {args.num_samples} samples...")
    samples = generate_samples(model, tokenizer, test_set, args.num_samples)

    # Score generations with RM if available
    if rm_model:
        logger.info("Scoring generated outputs with reward model...")
        samples = score_generations(rm_model, rm_tokenizer, rm_type, samples)

    for i, s in enumerate(samples):
        logger.info(f"\n--- Sample {i + 1} ---")
        if s["prompt"]:
            logger.info(f"Prompt: {s['prompt'][-1]['content'][:150]}...")
        logger.info(f"Generated: {s['generated'][:300]}...")
        logger.info(f"Chosen:    {s['chosen_start'][:300]}...")
        if "rm_score_generated" in s:
            logger.info(
                f"RM scores: generated={s['rm_score_generated']:.3f} "
                f"chosen={s['rm_score_chosen']:.3f} "
                f"delta={s['rm_score_delta']:.3f}"
            )

    results["samples"] = samples

    # --- 4. Best-of-N ---
    if args.best_of_n > 0:
        logger.info(f"Running best-of-{args.best_of_n} generation...")
        bon_results = best_of_n_generation(
            model, tokenizer, rm_model, rm_tokenizer, rm_type,
            test_set, n=args.best_of_n, num_samples=args.num_samples,
        )
        mean_spread = sum(r["score_spread"] for r in bon_results) / len(bon_results)
        logger.info(f"Best-of-{args.best_of_n} mean score spread: {mean_spread:.4f}")
        results["best_of_n"] = {
            "n": args.best_of_n,
            "mean_spread": mean_spread,
            "samples": bon_results,
        }

    # --- 5. Extra metrics (diversity / novelty / naturalness / tone) ---
    if args.extra_metrics:
        logger.info(
            f"Generating {args.diversity_k} samples x {args.diversity_prompts} "
            f"prompts for extra metrics..."
        )
        multi_groups = generate_multi_samples(
            model, tokenizer, test_set,
            num_prompts=args.diversity_prompts, k=args.diversity_k,
        )
        flat_gens = [t for g in multi_groups for t in g["generations"]]

        extra: dict = {}

        logger.info("Computing diversity metrics (distinct-n, self-BLEU)...")
        extra["diversity"] = compute_diversity_metrics(multi_groups)
        logger.info(
            f"  distinct_1={extra['diversity']['distinct_1']:.4f} "
            f"distinct_2={extra['diversity']['distinct_2']:.4f} "
            f"self_bleu={extra['diversity']['self_bleu']}"
        )

        # References for MAUVE and the training corpus for overlap are the
        # human assistant turns from the held-out test set and the train set.
        test_refs = [
            _assistant_text(ex["chosen"])
            for ex in test_set
            if _assistant_text(ex["chosen"])
        ]
        train_texts = [
            _assistant_text(ex["chosen"])
            for ex in dataset["train"]
            if _assistant_text(ex["chosen"])
        ]

        logger.info("Computing MAUVE (generations vs held-out human continuations)...")
        mauve_res = compute_mauve_score(flat_gens, test_refs)
        if mauve_res is not None:
            extra["mauve"] = mauve_res
            logger.info(f"  MAUVE: {mauve_res['mauve']:.4f}")

        logger.info(
            f"Computing training-set {args.novelty_ngram}-gram overlap "
            f"(up to {args.novelty_train_docs} docs)..."
        )
        extra["training_overlap"] = compute_training_overlap(
            flat_gens, train_texts,
            n=args.novelty_ngram, max_training_docs=args.novelty_train_docs,
        )
        logger.info(f"  overlap_rate={extra['training_overlap']['overlap_rate']:.4f}")

        logger.info("Computing formality scores (informal = higher is casual)...")
        form_res = compute_formality_scores(flat_gens)
        if form_res is not None:
            extra["formality"] = form_res
            logger.info(f"  mean_informal_prob={form_res['mean_informal_prob']:.4f}")

        results["extra_metrics"] = extra

    # --- Save ---
    if args.output:
        output_path = args.output
    elif args.baseline:
        tag = cfg.get("_model_override") or cfg["model"]["name"].replace("/", "_")
        output_path = os.path.join("outputs", f"baseline_{tag}", "eval_results.json")
    else:
        output_path = os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
