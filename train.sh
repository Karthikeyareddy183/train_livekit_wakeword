#!/usr/bin/env bash
# train.sh - one-command end-to-end training for hey_tara
#
# Does:
#   1. Verifies venv + GPU
#   2. Runs livekit-wakeword setup (idempotent - only downloads missing data)
#   3. Runs the full pipeline (generate -> augment -> train -> export)
#   4. Copies the exported ONNX to the project root for easy retrieval
#   5. Prints total elapsed time
#
# Usage:
#   ./train.sh                          # uses configs/hey_tara.yaml
#   ./train.sh configs/some_other.yaml  # use a different config
#
# Prereqs: venv activated, system deps installed, requirements.txt installed.
# See README.md sections 1-4 for first-time setup.

set -euo pipefail

CONFIG="${1:-configs/hey_tara.yaml}"
LOG_FILE="train_$(date +%Y%m%d_%H%M%S).log"
START=$(date +%s)

print_header() {
    echo
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# 1. Sanity check the environment
print_header "1/4  Environment check"

if ! command -v livekit-wakeword >/dev/null 2>&1; then
    echo "ERROR: livekit-wakeword CLI not found."
    echo "Activate your venv first:  source env/bin/activate"
    exit 1
fi

python - <<'PY' || exit 1
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. PyTorch fell back to CPU.")
    print("See README.md step 3 to install PyTorch with sm_120 (Blackwell) support.")
    sys.exit(1)
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  CUDA: {torch.version.cuda}")
print(f"  PyTorch: {torch.__version__}")
PY

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found at $CONFIG"
    exit 1
fi
echo "  Config: $CONFIG"

# 2. Setup (idempotent - skips downloads already present)
print_header "2/4  Dataset setup (idempotent)"
livekit-wakeword setup --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"

# 3. Full training pipeline
print_header "3/4  Training pipeline (generate -> augment -> train -> export)"
livekit-wakeword run "$CONFIG" 2>&1 | tee -a "$LOG_FILE"

# 4. Surface the output ONNX
print_header "4/4  Locating exported model"
MODEL_PATH=$(find output -type f -name '*.onnx' -printf '%T@ %p\n' 2>/dev/null \
             | sort -n | tail -n 1 | cut -d' ' -f2-)

if [[ -n "${MODEL_PATH:-}" ]]; then
    cp "$MODEL_PATH" ./hey_tara.onnx
    SIZE=$(du -h ./hey_tara.onnx | cut -f1)
    echo "  Source: $MODEL_PATH"
    echo "  Copied: ./hey_tara.onnx  ($SIZE)"
else
    echo "  WARNING: no .onnx found under output/ - pipeline may have failed"
    echo "  Check $LOG_FILE for details"
    exit 1
fi

# 5. Total elapsed
END=$(date +%s)
ELAPSED=$((END - START))
printf "\n  Total elapsed: %02d:%02d:%02d\n" \
       $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "  Full log: $LOG_FILE"
echo
