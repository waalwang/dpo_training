#!/usr/bin/env bash
# vast.ai startup script for DPO training
# Base image: pytorch/pytorch:2.5.1-cuda12.4.1 (runtime -- nvcc installed below)
set -euo pipefail

REPO_DIR="/workspace/dpo_training"
LOG="$REPO_DIR/onstart.log"

echo "[onstart] $(date) - starting" | tee -a "$LOG"

# ── 1. System deps ────────────────────────────────────────────────────────────
apt-get update -qq && apt-get install -y -qq git rsync cuda-toolkit-12-4 > /dev/null

# ── 2. Clone / pull repo ──────────────────────────────────────────────────────
# Set REPO_URL in the vast.ai template env vars, or replace below.
# rsync dataset:
#   rsync -avz -e "ssh -p <port>" \
#      <local path> \
#      root@<host>:<remote path>
REPO_URL="${REPO_URL:-}"
if [ -n "$REPO_URL" ]; then
    if [ -d "$REPO_DIR/.git" ]; then
        echo "[onstart] pulling latest" | tee -a "$LOG"
        git -C "$REPO_DIR" pull --ff-only
    else
        echo "[onstart] cloning $REPO_URL" | tee -a "$LOG"
        git clone "$REPO_URL" "$REPO_DIR"
    fi
else
    echo "[onstart] REPO_URL not set -- skipping clone (mount or copy manually)" | tee -a "$LOG"
fi

cd "$REPO_DIR"

# ── 3. Python deps ────────────────────────────────────────────────────────────
echo "[onstart] installing requirements" | tee -a "$LOG"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# deepspeed -- only install if cloud_full profile is planned
if [ "${INSTALL_DEEPSPEED:-0}" = "1" ]; then
    echo "[onstart] installing deepspeed" | tee -a "$LOG"
    pip install --quiet deepspeed>=0.14.0
fi

# flash-attn: pre-built wheel is much faster than compiling
# version must match torch+cuda combo: torch 2.5.1 + cuda 12.4
echo "[onstart] installing flash-attn" | tee -a "$LOG"
pip install --quiet flash-attn --no-build-isolation 2>>"$LOG" || \
    echo "[onstart] WARNING: flash-attn install failed -- training will fall back to eager" | tee -a "$LOG"

# ── 4. WandB login ────────────────────────────────────────────────────────────
# Set WANDB_API_KEY in vast.ai template env vars.
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "[onstart] logging in to wandb" | tee -a "$LOG"
    wandb login "$WANDB_API_KEY" --relogin > /dev/null
else
    echo "[onstart] WANDB_API_KEY not set -- wandb will be disabled" | tee -a "$LOG"
    export WANDB_DISABLED=true
fi

# ── 5. HuggingFace token (for gated models e.g. LLaMA) ───────────────────────
# Set HF_TOKEN in vast.ai template env vars.
if [ -n "${HF_TOKEN:-}" ]; then
    echo "[onstart] setting HF token" | tee -a "$LOG"
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential > /dev/null 2>&1 || true
fi

# ── 6. Smoke-test GPU ─────────────────────────────────────────────────────────
python - <<'EOF' | tee -a "$LOG"
import torch
print(f"[gpu] CUDA available: {torch.cuda.is_available()}")
print(f"[gpu] device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"[gpu] device {i}: {p.name} | {p.total_memory // 1024**3} GB")
EOF

echo "[onstart] $(date) - done" | tee -a "$LOG"
echo ""
echo "To start training:"
echo "  cd $REPO_DIR"
echo "  python train.py --profile cloud"
echo "  # or for 14B:"
echo "  python train.py --profile cloud_qlora_big --model-override qwen-14b"
echo "  # or for full FT (multi-GPU):"
echo "  accelerate launch --config_file configs/deepspeed_z3.yaml train.py --profile cloud_full --model-override qwen-7b"
