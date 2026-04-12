from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from audio_utils.wav_loader import DEFAULT_AUDIO_PATH, load_wav_cpu
from dali_pipeline.audio_decode import dali_available, decode_audio_with_dali


def benchmark_cpu(filepath: str | Path = DEFAULT_AUDIO_PATH, runs: int = 5) -> Tuple[np.ndarray, int, np.ndarray]:
    times = []
    last_audio = None
    last_sr = None
    for _ in range(runs):
        start = time.perf_counter()
        last_audio, last_sr = load_wav_cpu(filepath)
        times.append(time.perf_counter() - start)
    return last_audio, int(last_sr), np.asarray(times, dtype=np.float64)


def benchmark_dali(filepath: str | Path = DEFAULT_AUDIO_PATH, runs: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not dali_available():
        raise ImportError("DALI benchmark requested, but NVIDIA DALI is not available.")

    times = []
    last_audio = None
    last_sr = None
    for _ in range(runs):
        start = time.perf_counter()
        last_audio, last_sr = decode_audio_with_dali(filepath)
        times.append(time.perf_counter() - start)
    return np.asarray(last_audio), np.asarray(last_sr), np.asarray(times, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPU WAV decoding against DALI decoding.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_AUDIO_PATH), help="Path to input WAV file.")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    cpu_audio, cpu_sr, cpu_times = benchmark_cpu(input_path, runs=args.runs)

    print("\n===== Decode Benchmark Results =====")
    print(f"Input file              : {input_path}")
    print(f"CPU average decode time : {cpu_times.mean():.6f} s")
    print(f"CPU sample rate         : {cpu_sr}")
    print(f"CPU samples loaded      : {len(cpu_audio)}")

    if dali_available():
        dali_audio, dali_sr, dali_times = benchmark_dali(input_path, runs=args.runs)
        print(f"DALI average decode time: {dali_times.mean():.6f} s")
        print(f"DALI sample rate        : {int(np.asarray(dali_sr).reshape(-1)[0])}")
        print(f"DALI output shape       : {dali_audio.shape}")
    else:
        print("DALI average decode time: skipped (DALI not installed)")


if __name__ == "__main__":
    main()
