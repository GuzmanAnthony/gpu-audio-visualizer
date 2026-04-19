from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from audio_utils.features import compute_fft_bands, summarize_features
from audio_utils.wav_loader import DEFAULT_AUDIO_PATH, load_wav_cpu
from dali_pipeline.audio_decode import dali_available, decode_audio_with_dali


def benchmark_cpu_decode(filepath: Path, runs: int) -> Tuple[np.ndarray, int, np.ndarray]:
    times, audio, sr = [], None, None
    for _ in range(runs):
        t = time.perf_counter()
        audio, sr = load_wav_cpu(filepath)
        times.append(time.perf_counter() - t)
    return audio, int(sr), np.array(times)


def benchmark_dali_decode(filepath: Path, runs: int) -> Tuple[np.ndarray, int, np.ndarray]:
    if not dali_available():
        raise ImportError("DALI not available.")
    times, audio, sr = [], None, None
    for _ in range(runs):
        t = time.perf_counter()
        audio_batch, sr_batch = decode_audio_with_dali(filepath)
        times.append(time.perf_counter() - t)
    audio = np.asarray(audio_batch[0], dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    sr = int(np.asarray(sr_batch).reshape(-1)[0])
    return audio, sr, np.array(times)


# ── Full pipeline benchmarks (decode + FFT feature extraction) ─────────────────

def benchmark_cpu_full(filepath: Path, runs: int) -> Tuple[dict, np.ndarray]:
    times, features = [], None
    for _ in range(runs):
        t = time.perf_counter()
        audio, sr = load_wav_cpu(filepath)
        features = compute_fft_bands(audio, sr)
        times.append(time.perf_counter() - t)
    return features, np.array(times)


def benchmark_dali_full(filepath: Path, runs: int) -> Tuple[dict, np.ndarray]:
    if not dali_available():
        raise ImportError("DALI not available.")
    times, features = [], None
    for _ in range(runs):
        t = time.perf_counter()
        audio_batch, sr_batch = decode_audio_with_dali(filepath)
        audio = np.asarray(audio_batch[0], dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        sr = int(np.asarray(sr_batch).reshape(-1)[0])
        features = compute_fft_bands(audio, sr)
        times.append(time.perf_counter() - t)
    return features, np.array(times)


# ── Validation: confirm both pipelines produce equivalent features ─────────────

def validate_feature_parity(cpu_features: dict, dali_features: dict, tol: float = 1e-4) -> bool:
    """Check that CPU and DALI pipelines produce numerically equivalent features."""
    bands = ["bass", "mid", "treble", "level"]
    all_match = True
    for band in bands:
        cpu_arr  = np.array(cpu_features[band])
        dali_arr = np.array(dali_features[band])
        max_diff = np.abs(cpu_arr - dali_arr).max()
        match = max_diff < tol
        status = "OK" if match else "MISMATCH"
        print(f"  {band:<8} max diff: {max_diff:.2e}  [{status}]")
        if not match:
            all_match = False
    return all_match


# ── Report printer ─────────────────────────────────────────────────────────────

def _print_times(label: str, times: np.ndarray) -> None:
    print(f"  {label:<30} mean={times.mean():.4f}s  "
          f"min={times.min():.4f}s  max={times.max():.4f}s  "
          f"std={times.std():.4f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPU vs DALI decode and feature extraction.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_AUDIO_PATH))
    parser.add_argument("--runs",  type=int, default=5)
    args = parser.parse_args()

    filepath = Path(args.input).expanduser().resolve()
    print(f"\nInput : {filepath}")
    print(f"Runs  : {args.runs}\n")

    # ── Decode only ───────────────────────────────────────────────────────────
    print("===== DECODE ONLY =====")
    cpu_audio, cpu_sr, cpu_decode_times = benchmark_cpu_decode(filepath, args.runs)
    _print_times("CPU decode", cpu_decode_times)

    if dali_available():
        dali_audio, dali_sr, dali_decode_times = benchmark_dali_decode(filepath, args.runs)
        _print_times("DALI decode", dali_decode_times)
        speedup = cpu_decode_times.mean() / dali_decode_times.mean()
        print(f"  Decode speedup (DALI/CPU)     : {speedup:.2f}x")
    else:
        print("  DALI decode: skipped (not installed)")

    # ── Full pipeline ─────────────────────────────────────────────────────────
    print("\n===== FULL PIPELINE (decode + FFT bands) =====")
    cpu_features, cpu_full_times = benchmark_cpu_full(filepath, args.runs)
    _print_times("CPU full pipeline", cpu_full_times)

    if dali_available():
        dali_features, dali_full_times = benchmark_dali_full(filepath, args.runs)
        _print_times("DALI full pipeline", dali_full_times)
        speedup_full = cpu_full_times.mean() / dali_full_times.mean()
        print(f"  Full pipeline speedup (DALI/CPU): {speedup_full:.2f}x")

        # ── Parity check ──────────────────────────────────────────────────────
        print("\n===== FEATURE PARITY CHECK =====")
        ok = validate_feature_parity(cpu_features, dali_features)
        print(f"  Parity: {'PASS' if ok else 'FAIL'}")
    else:
        print("  DALI full pipeline: skipped (not installed)")

    # ── Feature summary ───────────────────────────────────────────────────────
    print("\n===== FEATURE SUMMARY (CPU pipeline) =====")
    print(f"  Frames : {len(cpu_features['times'])}")
    print(f"  SR     : {cpu_features['sr']} Hz")
    print(f"  Hop    : {cpu_features['hop']} samples")
    for band in ["bass", "mid", "treble", "level"]:
        arr = np.array(cpu_features[band])
        print(f"  {band:<8} mean={arr.mean():.4f}  max={arr.max():.4f}")


if __name__ == "__main__":
    main()