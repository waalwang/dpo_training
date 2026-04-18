"""
data_loader.py

Loads trajectory DPO parquet files from the crawler output and converts
them into the format expected by TRL's DPOTrainer.

DPOTrainer expects each row to have:
  - prompt:   list of {"role": ..., "content": ...} (the shared prefix)
  - chosen:   list of {"role": ..., "content": ...} (prefix + chosen trajectory)
  - rejected: list of {"role": ..., "content": ...} (prefix + rejected trajectory)

The crawler already produces this format, so we just parse the JSON strings
from parquet columns.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Optional

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


def load_trajectory_dataset(
    trajectory_dir: str,
    source: Optional[str] = None,
    test_split: float = 0.05,
    seed: int = 42,
    score_delta_threshold: float = 0.0,
) -> DatasetDict:
    """Load trajectory parquet files into a HuggingFace DatasetDict.

    Args:
        trajectory_dir: Path to directory with trajectory parquet shards.
        source: Filter by source ("reddit", "hacker_news", or None for all).
        test_split: Fraction to hold out for eval.
        seed: Random seed for the split.
        score_delta_threshold: Drop pairs with chosen-rejected score delta at or
            below this value. 0.0 keeps only strict positive deltas.

    Returns:
        DatasetDict with "train" and "test" splits.
    """
    pattern = os.path.join(trajectory_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {trajectory_dir}")

    logger.info(f"Found {len(files)} parquet shard(s) in {trajectory_dir}")

    # Load all shards
    raw_rows = []
    for fpath in files:
        basename = os.path.basename(fpath)
        # Filter by source if requested (files are named <source>_shard_XXXXX.parquet)
        if source and not basename.startswith(source):
            continue
        raw_rows.extend(_load_parquet_shard(fpath))

    if not raw_rows:
        raise ValueError(
            f"No rows loaded (source filter={source!r}). "
            f"Check that parquet files exist and match the source filter."
        )

    # Score-delta filter: drop pairs where chosen is not meaningfully better.
    before = len(raw_rows)
    rows = [
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in raw_rows
        if r["score_delta"] > score_delta_threshold
    ]
    dropped = before - len(rows)
    logger.info(
        f"Loaded {before} pairs | kept {len(rows)} (dropped {dropped} "
        f"with score_delta <= {score_delta_threshold}) | source={source or 'all'}"
    )
    if not rows:
        raise ValueError(
            f"All pairs filtered out by score_delta_threshold={score_delta_threshold}. "
            f"Lower the threshold or check the crawler output."
        )

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=test_split, seed=seed)
    logger.info(
        f"Split: train={len(split['train'])}, test={len(split['test'])}"
    )

    return split


def _strip_turns(turns: list[dict]) -> list[dict]:
    """Keep only role+content from turn dicts -- drop crawler metadata fields."""
    return [{"role": t["role"], "content": t["content"]} for t in turns]


def _load_parquet_shard(path: str) -> list[dict]:
    """Parse one parquet shard into DPO-ready dicts."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    data = table.to_pydict()
    n = table.num_rows

    rows = []
    for i in range(n):
        prompt = _strip_turns(json.loads(data["prompt"][i]))
        chosen = _strip_turns(json.loads(data["chosen"][i]))
        rejected = _strip_turns(json.loads(data["rejected"][i]))

        # TRL >= 0.9 concatenates prompt + chosen internally; store only the
        # response messages here to avoid double-prefixing.
        rows.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "score_delta": float(data["score_delta"][i]),
        })

    logger.info(f"  {os.path.basename(path)}: {n} pairs")
    return rows


def load_from_config(config: dict) -> DatasetDict:
    """Convenience: load dataset using the data section of a config dict."""
    data_cfg = config["data"]
    return load_trajectory_dataset(
        trajectory_dir=data_cfg["trajectory_dir"],
        source=data_cfg.get("source"),
        test_split=data_cfg.get("test_split", 0.05),
        seed=data_cfg.get("seed", 42),
        score_delta_threshold=data_cfg.get("score_delta_threshold", 0.0),
    )
