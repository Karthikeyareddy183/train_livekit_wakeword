"""Run the trained 'Hey Tara' model against a WAV file and report detections.

Slides a 2.0 s window (the training clip_duration) across the file with a 0.5 s stride
and prints the per-window score. Reports max score + any detections above threshold.

Usage:
    python test_wakeword.py path\\to\\audio.wav
    python test_wakeword.py path\\to\\audio.wav --threshold 0.4
    python test_wakeword.py path\\to\\audio.wav --verbose
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from livekit.wakeword import WakeWordModel

MODEL_PATH = Path("output") / "hey_tara" / "hey_tara.onnx"
TARGET_SR = 16000
WINDOW_SEC = 2.0
STRIDE_SEC = 0.5
WAKE_KEY = "hey_tara"


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)


def extract_score(scores, key: str) -> float:
    if isinstance(scores, dict):
        if key in scores:
            return float(scores[key])
        if len(scores) == 1:
            return float(next(iter(scores.values())))
        return 0.0
    if isinstance(scores, (int, float, np.floating)):
        return float(scores)
    arr = np.asarray(scores).ravel()
    return float(arr.max()) if arr.size else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    parser.add_argument("--stride-sec", type=float, default=STRIDE_SEC)
    parser.add_argument("--verbose", action="store_true",
                        help="Print every window's score, not just detections")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1

    audio = load_audio(args.wav)
    duration_s = len(audio) / TARGET_SR
    print(f"Loaded {args.wav}  duration={duration_s:.2f}s  samples={len(audio)}")

    window = int(args.window_sec * TARGET_SR)
    stride = int(args.stride_sec * TARGET_SR)
    if len(audio) < window:
        pad = np.zeros(window - len(audio), dtype=np.int16)
        audio = np.concatenate([audio, pad])

    model = WakeWordModel(models=[args.model])

    print(f"Sliding {args.window_sec}s window with {args.stride_sec}s stride")
    max_score = 0.0
    max_t = 0.0
    hits: list[tuple[float, float]] = []
    n_windows = 0
    for i in range(0, len(audio) - window + 1, stride):
        chunk = audio[i : i + window]
        score = extract_score(model.predict(chunk), WAKE_KEY)
        t = i / TARGET_SR
        n_windows += 1
        if score > max_score:
            max_score, max_t = score, t
        if args.verbose:
            print(f"  t={t:6.2f}s  score={score:.4f}")
        if score >= args.threshold:
            hits.append((t, score))
            if not args.verbose:
                print(f"  detection @ {t:6.2f}s   score={score:.3f}")

    print(f"\nWindows scored : {n_windows}")
    print(f"Max score      : {max_score:.4f}  (@ {max_t:.2f}s)")
    print(f"Detections >= {args.threshold} : {len(hits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
