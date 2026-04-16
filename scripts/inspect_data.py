"""
scripts/inspect_data.py

Quick sanity check on the trajectory dataset before training.
Shows stats, token length distributions, and sample pairs.

Usage:
    python scripts/inspect_data.py
    python scripts/inspect_data.py --source hacker_news --show-samples 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Inspect trajectory dataset")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None, help="Filter by source")
    parser.add_argument("--show-samples", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.source:
        cfg["data"]["source"] = args.source

    dataset = load_from_config(cfg)

    for split_name, split in dataset.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Split: {split_name} ({len(split)} examples)")
        logger.info(f"{'='*60}")

        if len(split) == 0:
            continue

        prompt_lens = []
        chosen_lens = []
        rejected_lens = []

        for ex in split:
            prompt_lens.append(len(ex["prompt"]))
            chosen_lens.append(len(ex["chosen"]))
            rejected_lens.append(len(ex["rejected"]))

        _print_distribution("Prompt turns", prompt_lens)
        _print_distribution("Chosen turns", chosen_lens)
        _print_distribution("Rejected turns", rejected_lens)

        # Token-level stats (approximate with char count / 4)
        prompt_chars = [sum(len(m["content"]) for m in ex["prompt"]) for ex in split]
        chosen_chars = [sum(len(m["content"]) for m in ex["chosen"]) for ex in split]
        rejected_chars = [sum(len(m["content"]) for m in ex["rejected"]) for ex in split]

        logger.info("\nApprox token counts (chars/4):")
        _print_distribution("  Prompt tokens", [c // 4 for c in prompt_chars])
        _print_distribution("  Chosen tokens", [c // 4 for c in chosen_chars])
        _print_distribution("  Rejected tokens", [c // 4 for c in rejected_chars])

        # Show samples
        if args.show_samples > 0:
            logger.info(f"\n--- Samples from {split_name} ---")
            for i in range(min(args.show_samples, len(split))):
                ex = split[i]
                logger.info(f"\n[Example {i + 1}]")
                logger.info(f"  Prompt ({len(ex['prompt'])} turns):")
                for turn in ex["prompt"][:3]:
                    content = turn["content"][:120]
                    logger.info(f"    [{turn['role']}] {content}...")
                if len(ex["prompt"]) > 3:
                    logger.info(f"    ... ({len(ex['prompt']) - 3} more turns)")

                chosen_traj = ex["chosen"][len(ex["prompt"]):]
                rejected_traj = ex["rejected"][len(ex["prompt"]):]
                logger.info(f"  Chosen trajectory ({len(chosen_traj)} turns):")
                if chosen_traj:
                    logger.info(f"    [{chosen_traj[0]['role']}] {chosen_traj[0]['content'][:120]}...")
                logger.info(f"  Rejected trajectory ({len(rejected_traj)} turns):")
                if rejected_traj:
                    logger.info(f"    [{rejected_traj[0]['role']}] {rejected_traj[0]['content'][:120]}...")


def _print_distribution(label: str, values: list[int]):
    if not values:
        logger.info(f"  {label}: (empty)")
        return
    values_sorted = sorted(values)
    n = len(values_sorted)
    logger.info(
        f"  {label}: "
        f"min={values_sorted[0]} "
        f"p25={values_sorted[n // 4]} "
        f"p50={values_sorted[n // 2]} "
        f"p75={values_sorted[3 * n // 4]} "
        f"max={values_sorted[-1]} "
        f"mean={sum(values) / n:.1f}"
    )


if __name__ == "__main__":
    main()
