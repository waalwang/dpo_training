"""
train.py

DPO fine-tuning on trajectory preference data.
Supports QLoRA (default) and full fine-tuning.
Supports two-stage training: SFT on chosen trajectories, then DPO/IPO.

Usage:
    # Local QLoRA, Qwen2.5-1.5B (TITAN RTX 24GB):
    python train.py --profile local

    # Cloud QLoRA, Qwen2.5-1.5B (1xA100 40GB):
    python train.py --profile cloud

    # Cloud QLoRA, Qwen2.5-14B (1xA100 80GB):
    python train.py --profile cloud_qlora_big --model-override qwen-14b

    # Cloud full fine-tune, Qwen2.5-7B (2xA100 80GB, DeepSpeed ZeRO-3):
    accelerate launch --config_file configs/deepspeed_z3.yaml \
        train.py --profile cloud_full --model-override qwen-7b

    # Cloud full fine-tune, LLaMA 3.1-8B:
    accelerate launch --config_file configs/deepspeed_z3.yaml \
        train.py --profile cloud_full --model-override llama-8b

    # Two-stage: SFT then DPO in one run:
    python train.py --profile cloud --stage both

    # SFT only (saves to outputs/sft/final):
    python train.py --profile cloud --stage sft

    # DPO only, starting from a saved SFT checkpoint:
    python train.py --profile cloud --stage dpo --sft-checkpoint outputs/sft/final

    # Dry run (load data + model, skip training):
    python train.py --profile local --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import yaml
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from data_loader import load_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str, profile: str, model_override: str | None = None) -> dict:
    """Load YAML config, merge hardware profile, and apply model override."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    hw = cfg["profiles"][profile]
    cfg["training"].update(hw)
    cfg["_profile"] = profile

    if model_override:
        presets = cfg.get("model_presets", {})
        if model_override not in presets:
            raise ValueError(
                f"Unknown model preset: {model_override!r}. "
                f"Available: {list(presets.keys())}"
            )
        cfg["model"]["name"] = presets[model_override]["name"]
        logger.info(f"Model override: {model_override} -> {cfg['model']['name']}")

    cfg["_full_finetune"] = cfg["training"].get("full_finetune", False)
    return cfg


def build_quantization_config(cfg: dict) -> BitsAndBytesConfig:
    """4-bit NF4 quantization for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg["training"]["bf16"] else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    """Build LoRA config from the qlora section."""
    qlora = cfg["qlora"]
    return LoraConfig(
        r=qlora["r"],
        lora_alpha=qlora["lora_alpha"],
        lora_dropout=qlora["lora_dropout"],
        target_modules=qlora["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def build_training_args(cfg: dict) -> DPOConfig:
    """Build DPOConfig (extends TrainingArguments) from config."""
    t = cfg["training"]
    return DPOConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        lr_scheduler_type=t["lr_scheduler_type"],
        bf16=t["bf16"],
        fp16=t["fp16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy=t.get("eval_strategy", "steps"),
        save_total_limit=t["save_total_limit"],
        report_to=t.get("report_to", "none"),
        run_name=t.get("run_name"),
        beta=t["beta"],
        loss_type=t["loss_type"],
        rpo_alpha=t.get("rpo_alpha"),
        max_length=t["max_length"],
        remove_unused_columns=False,
        seed=cfg["data"].get("seed", 42),
    )


def load_model_and_tokenizer(cfg: dict):
    """Load the base model and tokenizer.

    QLoRA profiles: 4-bit quantized, prepare_model_for_kbit_training.
    Full fine-tune profiles: full precision (bf16/fp16), no quantization.
    """
    model_name = cfg["model"]["name"]
    t = cfg["training"]
    full_ft = cfg["_full_finetune"]

    logger.info(f"Loading model: {model_name}")
    logger.info(
        f"Profile: {cfg['_profile']} | full_finetune={full_ft} | "
        f"bf16={t['bf16']} | fp16={t['fp16']}"
    )

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if full_ft:
        # Full precision for full fine-tune (bf16 handled by trainer)
        if t["bf16"]:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif t["fp16"]:
            model_kwargs["torch_dtype"] = torch.float16
    else:
        # QLoRA: 4-bit quantization
        model_kwargs["quantization_config"] = build_quantization_config(cfg)

    # Flash attention 2 for Ampere+ GPUs
    attn_impl = t.get("attn_implementation")
    if attn_impl and attn_impl != "eager":
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if not full_ft:
        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for DPO

    return model, tokenizer


def build_sft_args(cfg: dict) -> SFTConfig:
    """Build SFTConfig from the sft section of config, falling back to training defaults."""
    t   = cfg["training"]
    sft = cfg.get("sft", {})
    return SFTConfig(
        output_dir=sft.get("output_dir", os.path.join(t["output_dir"], "sft")),
        num_train_epochs=sft.get("num_train_epochs", 1),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=sft.get("learning_rate", t["learning_rate"]),
        warmup_steps=sft.get("warmup_steps", t["warmup_steps"]),
        weight_decay=t["weight_decay"],
        lr_scheduler_type=t["lr_scheduler_type"],
        bf16=t["bf16"],
        fp16=t["fp16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=sft.get("eval_steps", t["eval_steps"]),
        eval_strategy=sft.get("eval_strategy", "steps"),
        save_total_limit=t["save_total_limit"],
        report_to=t.get("report_to", "none"),
        run_name=(t.get("run_name") or "") + "-sft",
        max_length=t["max_length"],
        seed=cfg["data"].get("seed", 42),
    )


def run_sft_stage(
    cfg: dict,
    model,
    tokenizer,
    dataset: DatasetDict,
    resume_from_checkpoint: str | None = None,
) -> str:
    """Stage 1: SFT on chosen trajectories.

    Each chosen entry is a full conversation (prefix + chosen turns) as a
    list of message dicts. SFTTrainer applies the chat template and supervises
    on all assistant tokens.

    Returns the path to the saved SFT checkpoint.
    """
    # SFTTrainer expects a 'messages' column (list of dicts) or a 'text' column.
    # Our 'chosen' field is already prompt+chosen turns as message dicts.
    sft_train = dataset["train"].select_columns(["chosen"]).rename_column("chosen", "messages")
    sft_eval  = dataset["test"].select_columns(["chosen"]).rename_column("chosen", "messages")

    sft_args = build_sft_args(cfg)
    trainer  = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=sft_train,
        eval_dataset=sft_eval,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT stage...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = os.path.join(sft_args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"SFT checkpoint saved to {final_dir}")
    return final_dir


def main():
    parser = argparse.ArgumentParser(description="DPO training on trajectory data")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--profile", default="local",
        choices=["local", "cloud", "cloud_full", "cloud_qlora_big"],
        help="Hardware profile to use",
    )
    parser.add_argument(
        "--model-override", default=None,
        help="Model preset key (qwen-1.5b, qwen-7b, qwen-14b, llama-8b)",
    )
    parser.add_argument(
        "--stage", default="dpo", choices=["sft", "dpo", "both"],
        help="Training stage: sft, dpo, or both (sft then dpo)",
    )
    parser.add_argument(
        "--sft-checkpoint", default=None,
        help="Path to SFT checkpoint to initialise the DPO stage from. "
             "Only used when --stage=dpo.",
    )
    parser.add_argument(
        "--resume-from-checkpoint", default=None,
        help="Path to a checkpoint to resume training from (works for both SFT and DPO stages).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load data and model, print stats, skip training",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.profile, args.model_override)
    full_ft = cfg["_full_finetune"]

    # --- Data ---
    logger.info("Loading trajectory dataset...")
    dataset = load_from_config(cfg)

    if args.dry_run:
        _print_data_stats(dataset)

    # --- Model ---
    # For dpo-only stage, optionally load from a saved SFT checkpoint.
    if args.stage == "dpo" and args.sft_checkpoint:
        logger.info(f"Loading SFT checkpoint for DPO stage: {args.sft_checkpoint}")
        cfg["model"]["name"] = args.sft_checkpoint

    model, tokenizer = load_model_and_tokenizer(cfg)

    total = sum(p.numel() for p in model.parameters())

    # --- LoRA (skip for full fine-tune) ---
    if full_ft:
        logger.info(f"Full fine-tune: {total:,} params, all trainable")
    else:
        lora_config = build_lora_config(cfg)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return

    # --- Stage 1: SFT ---
    if args.stage in ("sft", "both"):
        sft_checkpoint = run_sft_stage(cfg, model, tokenizer, dataset,
                                       resume_from_checkpoint=args.resume_from_checkpoint)
        if args.stage == "sft":
            return
        # Reload the SFT-trained weights for DPO stage
        logger.info("Reloading SFT checkpoint for DPO stage...")
        cfg["model"]["name"] = sft_checkpoint
        model, tokenizer = load_model_and_tokenizer(cfg)
        if not full_ft:
            model = get_peft_model(model, build_lora_config(cfg))

    # --- Stage 2: DPO ---
    training_args = build_training_args(cfg)

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training...")
    dpo_resume = args.resume_from_checkpoint if args.stage == "dpo" else None
    trainer.train(resume_from_checkpoint=dpo_resume)

    # --- Save ---
    final_dir = os.path.join(cfg["training"]["output_dir"], "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Model saved to {final_dir}")


def _print_data_stats(dataset: DatasetDict):
    """Print dataset statistics for dry-run inspection."""
    for split_name, split in dataset.items():
        logger.info(f"--- {split_name} split: {len(split)} examples ---")
        if len(split) == 0:
            continue
        sample = split[0]
        logger.info(f"  prompt turns:   {len(sample['prompt'])}")
        logger.info(f"  chosen turns:   {len(sample['chosen'])}")
        logger.info(f"  rejected turns: {len(sample['rejected'])}")
        # Show first prompt turn
        if sample["prompt"]:
            content = sample["prompt"][0].get("content", "")
            logger.info(f"  first prompt:   {content[:120]}...")


if __name__ == "__main__":
    main()
