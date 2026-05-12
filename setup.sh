#!/usr/bin/env bash
# setup.sh - one-time environment prep for the Ubuntu / RTX 5090 box
#
# Does:
#   1. Installs Ubuntu system deps (espeak-ng, ffmpeg, sox, libsndfile1, portaudio)
#   2. Creates a Python 3.12 venv at ./env
#   3. Installs PyTorch with CUDA 12.8+ (Blackwell / sm_120)
#   4. Installs livekit-wakeword + audio I/O libs from requirements.txt
#   5. Verifies the GPU is recognized by PyTorch
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# After it finishes:
#   source env/bin/activate
#   ./train.sh
#
# Re-running is safe - all steps are idempotent.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

print_header() {
    echo
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# 1. System deps
print_header "1/5  System dependencies (apt)"
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y \
        espeak-ng libsndfile1 ffmpeg sox portaudio19-dev
else
    echo "WARNING: apt-get not found. Install these manually if missing:"
    echo "  espeak-ng libsndfile1 ffmpeg sox portaudio19-dev"
fi

# 2. Python version check
print_header "2/5  Python version check"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN not found."
    echo "Install Python 3.11 or 3.12 (NOT 3.13 - webrtcvad has no wheel for it):"
    echo "  sudo apt install python3.12 python3.12-venv"
    exit 1
fi
"$PYTHON_BIN" --version

# 3. venv
print_header "3/5  Python venv (./env)"
if [[ ! -d env ]]; then
    "$PYTHON_BIN" -m venv env
    echo "  Created ./env"
else
    echo "  ./env already exists - reusing"
fi
# shellcheck source=/dev/null
source env/bin/activate
python -m pip install --upgrade pip --quiet

# 4. PyTorch with CUDA 12.8 (Blackwell sm_120 support)
print_header "4/5  PyTorch with CUDA 12.8+"
echo "  Installing torch torchvision torchaudio from cu128 index..."
pip install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Verify GPU is recognized; fall back to nightly if not
if ! python - <<'PY' 2>/dev/null
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
then
    echo "  Stable cu128 build did not detect GPU. Falling back to nightly cu128..."
    pip install --quiet --upgrade --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128
fi

python - <<'PY'
import torch
print(f"  PyTorch:    {torch.__version__}")
print(f"  CUDA:       {torch.version.cuda}")
print(f"  CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU:        {torch.cuda.get_device_name(0)}")
PY

# 5. livekit-wakeword + audio libs
print_header "5/5  livekit-wakeword + audio libs"
pip install -r requirements.txt
livekit-wakeword --help >/dev/null && echo "  livekit-wakeword CLI is callable"

echo
echo "============================================================"
echo "  Setup complete."
echo "============================================================"
echo
echo "Next steps:"
echo "  1. (optional) Drop kitchen audio into ./data/backgrounds/kitchen_custom/"
echo "  2. source env/bin/activate"
echo "  3. ./train.sh"
echo
