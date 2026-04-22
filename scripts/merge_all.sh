#!/usr/bin/env bash
# Merge every adapter checkpoint under outputs/<run>/checkpoint-* into
# merged/<run>/checkpoint-*. Skips runs listed in SKIP_RUNS and output dirs
# that already exist.
#
# Usage:
#     ./scripts/merge_all.sh                # merge all runs
#     RUNS="dpo_clean_probe_v1" ./scripts/merge_all.sh   # merge one run
#     DTYPE=fp16 ./scripts/merge_all.sh     # override dtype

set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
OUTPUTS_DIR="${OUTPUTS_DIR:-outputs}"
MERGED_DIR="${MERGED_DIR:-merged}"
DTYPE="${DTYPE:-bf16}"
SKIP_RUNS="${SKIP_RUNS:-sft dpo_beta_5e-2}"

# Allow caller to pin specific runs; otherwise enumerate outputs/
if [ -n "${RUNS:-}" ]; then
    run_dirs=()
    for r in $RUNS; do run_dirs+=("$OUTPUTS_DIR/$r"); done
else
    run_dirs=("$OUTPUTS_DIR"/*/)
fi

for run_dir in "${run_dirs[@]}"; do
    run_dir="${run_dir%/}"
    run_name="$(basename "$run_dir")"

    # skip non-dirs and skip-list entries
    [ -d "$run_dir" ] || continue
    case " $SKIP_RUNS " in *" $run_name "*) echo "skip $run_name (in SKIP_RUNS)"; continue;; esac

    for ckpt in "$run_dir"/checkpoint-*; do
        [ -d "$ckpt" ] || continue
        [ -f "$ckpt/adapter_config.json" ] || { echo "skip $ckpt (no adapter_config.json)"; continue; }

        ckpt_name="$(basename "$ckpt")"
        out="$MERGED_DIR/$run_name/$ckpt_name"

        if [ -d "$out" ]; then
            echo "skip $out (exists)"
            continue
        fi

        echo ">>> merging $ckpt -> $out"
        "$PY" scripts/merge_adapter.py --adapter "$ckpt" --out "$out" --dtype "$DTYPE"
    done
done

echo "done."
