# train_livekit_wakeword

Training the `Hey Tara` wake word on Ubuntu + RTX 5090. End-to-end the first run takes about **1.5–2 hours**: ~30–45 min for the one-time dataset download, ~45–75 min for actual training. Subsequent runs (no re-download) take ~45–75 min.

---

## Quick start (two commands)

After cloning, on the Ubuntu / RTX 5090 box:

```bash
# one-time environment prep
chmod +x setup.sh train.sh
./setup.sh                    # apt deps + Python venv + PyTorch (cu128) + livekit-wakeword

# every training run
source env/bin/activate
./train.sh                    # setup datasets (idempotent) + generate + augment + train + export
```

When `train.sh` finishes, the trained model is at `./hey_tara.onnx`. Full per-run log saved to `train_<timestamp>.log`.

| Script | When | What it does |
|---|---|---|
| `setup.sh` | **Once** on a fresh machine | apt system deps → Python 3.12 venv → PyTorch cu128 (auto-falls-back to nightly if needed) → installs `requirements.txt` → verifies GPU |
| `train.sh` | **Every** training run | `livekit-wakeword setup` (idempotent) → `livekit-wakeword run` → copies the exported ONNX to `./hey_tara.onnx` |

The detailed step-by-step below is the manual equivalent of what these scripts do — useful for debugging or running individual stages.

---

## 0. Prerequisites

- Ubuntu 22.04 or 24.04
- NVIDIA driver supporting RTX 5090 (Blackwell, sm_120) — driver version **≥ 570**
- Python **3.11 or 3.12** (do NOT use 3.13 — `webrtcvad`, a transitive dep, has no prebuilt wheel for 3.13 and source-build is brittle)
- ~50 GB free disk for the dataset cache under `./data/`
- Internet (one-time dataset download is ~30 GB)

Verify the driver:

```bash
nvidia-smi
```

Should show `NVIDIA GeForce RTX 5090` and CUDA Version 12.8 or higher.

---

## 1. Install system dependencies

```bash
sudo apt update
sudo apt install -y espeak-ng libsndfile1 ffmpeg sox portaudio19-dev
```

- `espeak-ng` — phoneme generation for the Piper TTS backend
- `libsndfile1` — backs `soundfile` for WAV I/O
- `ffmpeg` + `sox` — audio resampling / format conversion
- `portaudio19-dev` — only needed if you also want to run the microphone listener locally

---

## 2. Create a Python venv

```bash
cd /path/to/Livekit-wakeword
python3.12 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
```

---

## 3. Install PyTorch with Blackwell (sm_120) support

The 5090 needs CUDA 12.8+ kernels. Stable PyTorch (as of late 2025) may or may not ship sm_120 — check first, fall back to nightly if needed.

```bash
# Try stable first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify the GPU is recognized:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True NVIDIA GeForce RTX 5090
```

If you see `False`, or get a `no kernel image is available for execution on the device` error during training, install the nightly build instead:

```bash
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## 4. Install LiveKit-wakeword + training extras

```bash
pip install -r requirements.txt
```

This pulls `livekit-wakeword[train,eval,export]` plus `soundfile`, `numpy`, `librosa`.

Sanity check:

```bash
livekit-wakeword --help
```

---

## 5. Download datasets (one-time, ~30 GB)

```bash
livekit-wakeword setup --config configs/hey_tara.yaml
```

This pulls into `./data/`:
- **MUSAN noise** (~6 hours of background noise) → `./data/backgrounds/`
- **MIT environmental impulse responses** (~270 room reverbs) → `./data/rirs/`
- **ACAV100M precomputed features** (~16 GB of random-speech embeddings) → `./data/`

Safe to interrupt and re-run — it's idempotent.

---

## 6. (Recommended) Add kitchen-specific background audio

The default MUSAN backgrounds are general-purpose. For a kitchen-deployed device, add ~60–100 WAVs of real kitchen sounds so the model learns to reject them.

```bash
mkdir -p ./data/backgrounds/kitchen_custom
# copy / scp your kitchen recordings into ./data/backgrounds/kitchen_custom/
```

**Format required:** 16 kHz, mono, WAV. To convert:

```bash
ffmpeg -i raw_recording.m4a -ar 16000 -ac 1 ./data/backgrounds/kitchen_custom/kitchen_sizzle.wav
```

**What to cover** (~1–3 hours of audio total, split into 30 s – 3 min files):

- Water: tap, kettle, dishwasher
- Cooking: sizzling, frying, boiling, oven beeps
- Appliances: blender, mixer, microwave, range hood, coffee grinder
- Impacts: chopping, pot/pan clanging, cutlery
- Background talk: family chatter, kids, kitchen radio/TV

The augmenter recursively scans `./data/backgrounds/`, so no config change is needed beyond keeping `background_paths: [./data/backgrounds]` in the YAML (already set).

**Do not** include any clip where someone says "Hey Tara" or anything close to it — it would poison the negative set.

---

## 7. Review the config

Open `configs/hey_tara.yaml`. Key knobs already tuned for the 5090:

| Field | Value | Why |
|---|---|---|
| `n_samples` | 25000 | Production-scale positives + adversarials |
| `model.model_size` | `medium` (128-dim) | Plenty of VRAM headroom |
| `steps` | 60000 | Solid convergence; bump to 100000 for max quality |
| `augmentation.rounds` | 3 | Heavy acoustic variety |
| `tts_batch_size` | 100 | Fills VRAM during TTS gen |
| `ACAV100M_sample` | 2048 | Large random-speech negative pool per step |
| `target_fp_per_hour` | 0.1 | Tight FP target |

For faster iteration during development, drop to:
- `n_samples: 10000`, `steps: 30000`, `model_size: small` → ~20–30 min instead of ~60 min.

---

## 8. Train

```bash
livekit-wakeword run configs/hey_tara.yaml
```

This runs the full pipeline:
1. `generate` — TTS-synthesize positives + adversarial negatives
2. `augment` — apply RIR convolution, background mixing, gain perturbation (× `rounds`)
3. `train` — 60000 steps of conv-attention classifier
4. `export` — emit ONNX

Approximate timing on RTX 5090:

| Phase | Time |
|---|---|
| Generate | 5–10 min |
| Augment | 5–10 min |
| Train (60k steps, medium) | 30–50 min |
| Export | < 1 min |
| **Total** | **~45–75 min** |

You can run sub-stages individually if you need to debug a single phase:

```bash
livekit-wakeword generate configs/hey_tara.yaml
livekit-wakeword augment  configs/hey_tara.yaml
livekit-wakeword train    configs/hey_tara.yaml
livekit-wakeword export   configs/hey_tara.yaml
livekit-wakeword eval     configs/hey_tara.yaml
```

---

## 9. Output

The exported model lands at:

```
./output/hey_tara/hey_tara.onnx
```

Also produced:
- `./output/hey_tara/checkpoint_*.pt` — PyTorch checkpoints
- `./output/hey_tara/eval_report.*` — validation metrics, threshold-tuning info
- `./output/hey_tara/synthetic_data/` — generated audio (can delete to reclaim disk)

---

## 10. Move the model to the deployment target

Copy just the ONNX. Inference is light and runs anywhere `onnxruntime` runs (laptop, Raspberry Pi 5, etc.):

```bash
# example: SCP back to the laptop
scp ./output/hey_tara/hey_tara.onnx user@laptop:/path/to/Livekit-wakeword/
```

On the deployment box, run the included `test_wakeword.py` to verify against a WAV file:

```bash
python test_wakeword.py path/to/audio.wav --model hey_tara.onnx
```

---

## Troubleshooting

**`webrtcvad` fails to build during pip install.**
Python 3.13 has no prebuilt wheel. Recreate the venv on Python 3.11 or 3.12.

**`torch.cuda.is_available()` is `False` despite `nvidia-smi` working.**
PyTorch doesn't have the right CUDA build for sm_120. Reinstall using the nightly `cu128` index (step 3).

**`no kernel image is available for execution on the device` during training.**
Same root cause as above — install nightly PyTorch with CUDA 12.8+.

**OOM during TTS generation.**
Lower `tts_batch_size` from 100 to 50, or 32.

**OOM during training (unlikely with 24 GB).**
Lower `ACAV100M_sample` from 2048 to 1024, or `batch_n_per_class.positive` and `adversarial_negative` from 50 to 32.

**Training appears stuck after `generate`.**
The first augmentation round loads and convolves RIRs — first batch can take 2–4 min before progress prints resume. Subsequent rounds are much faster.

**False positives are still too high after training.**
Add more kitchen background audio (step 6), bump `max_negative_weight` from 3000 to 5000, and/or lower `target_fp_per_hour` from 0.1 to 0.05.

**Detections are weak (peaks below 0.5) for your specific voice.**
Wait for PR [#69](https://github.com/livekit/livekit-wakeword/pull/69) to merge, then add 30–60 real recordings of yourself saying "Hey Tara" via `custom_positive_samples`.
