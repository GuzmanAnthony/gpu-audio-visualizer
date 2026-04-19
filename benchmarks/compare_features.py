from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from audio_utils.features import DEFAULT_FRAME_SIZE, DEFAULT_HOP_SIZE, DEFAULT_WAVEFORM_BUCKETS, compute_feature_bundle_cpu
from audio_utils.wav_loader import DEFAULT_AUDIO_PATH, load_wav_cpu
from gpu_features import compute_feature_bundle_gpu


def _stats(times_ms: list[float]) -> Dict[str, float]:
    return {
        "mean_ms": float(statistics.mean(times_ms)),
        "median_ms": float(statistics.median(times_ms)),
        "std_ms": float(statistics.pstdev(times_ms)) if len(times_ms) > 1 else 0.0,
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
    }


def _run_cpu(signal: np.ndarray, sample_rate: int, frame_size: int, hop_size: int, waveform_buckets: int, runs: int, warmup: int) -> tuple[dict[str, Any], list[float]]:
    last_bundle = {}
    for _ in range(max(0, warmup)):
        last_bundle = compute_feature_bundle_cpu(signal, sample_rate, frame_size, hop_size, waveform_buckets)

    times_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        last_bundle = compute_feature_bundle_cpu(signal, sample_rate, frame_size, hop_size, waveform_buckets)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return last_bundle, times_ms


def _run_gpu(signal: np.ndarray, sample_rate: int, frame_size: int, hop_size: int, waveform_buckets: int, runs: int, warmup: int) -> tuple[dict[str, Any], list[float], dict[str, float]]:
    last_bundle = {}
    for _ in range(max(0, warmup)):
        last_bundle = compute_feature_bundle_gpu(signal, sample_rate, frame_size, hop_size, waveform_buckets)

    times_ms: list[float] = []
    kernel_components: dict[str, list[float]] = {}
    for _ in range(runs):
        start = time.perf_counter()
        last_bundle = compute_feature_bundle_gpu(signal, sample_rate, frame_size, hop_size, waveform_buckets)
        times_ms.append((time.perf_counter() - start) * 1000.0)
        for key, value in last_bundle.get("timings_ms", {}).items():
            kernel_components.setdefault(key, []).append(float(value))

    kernel_summary = {key: float(statistics.mean(values)) for key, values in kernel_components.items()}
    return last_bundle, times_ms, kernel_summary


def _max_abs_diff(cpu_bundle: dict[str, Any], gpu_bundle: dict[str, Any]) -> dict[str, float]:
    diffs: dict[str, float] = {}
    for section in ("frames", "waveform"):
        for key, cpu_values in cpu_bundle[section].items():
            gpu_values = gpu_bundle[section][key]
            if len(cpu_values) == 0 and len(gpu_values) == 0:
                diffs[f"{section}.{key}"] = 0.0
                continue
            if len(cpu_values) != len(gpu_values):
                diffs[f"{section}.{key}"] = float("inf")
                continue
            diffs[f"{section}.{key}"] = float(np.max(np.abs(np.asarray(cpu_values) - np.asarray(gpu_values))))
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU NumPy feature extraction against the CUDA backend.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_AUDIO_PATH), help="Input WAV file.")
    parser.add_argument("--runs", type=int, default=10, help="Number of timed runs after warmup.")
    parser.add_argument("--warmup", type=int, default=2, help="Number of untimed warmup runs.")
    parser.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE)
    parser.add_argument("--hop-size", type=int, default=DEFAULT_HOP_SIZE)
    parser.add_argument("--waveform-buckets", type=int, default=DEFAULT_WAVEFORM_BUCKETS)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    signal, sample_rate = load_wav_cpu(input_path)

    cpu_bundle, cpu_times = _run_cpu(signal, sample_rate, args.frame_size, args.hop_size, args.waveform_buckets, args.runs, args.warmup)
    gpu_bundle, gpu_times, gpu_kernel_breakdown = _run_gpu(signal, sample_rate, args.frame_size, args.hop_size, args.waveform_buckets, args.runs, args.warmup)

    cpu_stats = _stats(cpu_times)
    gpu_stats = _stats(gpu_times)
    speedup = cpu_stats["mean_ms"] / gpu_stats["mean_ms"] if gpu_stats["mean_ms"] > 0 else float("inf")
    diffs = _max_abs_diff(cpu_bundle, gpu_bundle)

    report = {
        "input": str(input_path),
        "sample_rate": int(sample_rate),
        "num_samples": int(signal.size),
        "cpu_feature_timing_ms": cpu_stats,
        "gpu_feature_timing_ms": gpu_stats,
        "gpu_kernel_breakdown_ms": gpu_kernel_breakdown,
        "speedup_cpu_over_gpu": float(speedup),
        "max_abs_diff": diffs,
    }

    print("==== CPU vs GPU Feature Benchmark ====")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
