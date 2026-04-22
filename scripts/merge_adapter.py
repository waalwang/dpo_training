"""Merge a PEFT/LoRA adapter checkpoint into its base model and save as a
plain HF model. Strips the bitsandbytes dependency so the merged folder
loads on any torch/cuda combo.

Usage:
    python scripts/merge_adapter.py \
        --adapter outputs/dpo_clean_probe_v1/checkpoint-5000 \
        --out merged/ekc33rny-ckpt5000

Optional:
    --base  override base model repo (default: read from adapter_config.json)
    --dtype bf16 | fp16 | fp32 (default: bf16)
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to adapter checkpoint dir")
    ap.add_argument("--out", required=True, help="output dir for merged model")
    ap.add_argument("--base", default=None, help="override base model repo id")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    base_id = args.base
    if base_id is None:
        with open(os.path.join(args.adapter, "adapter_config.json")) as f:
            base_id = json.load(f)["base_model_name_or_path"]
    print(f"base: {base_id}")
    print(f"adapter: {args.adapter}")
    print(f"dtype: {args.dtype}")

    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_id).save_pretrained(args.out)
    print(f"merged -> {args.out}")


if __name__ == "__main__":
    main()
