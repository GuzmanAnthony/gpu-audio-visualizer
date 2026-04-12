from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audio_utils.features import summarize_features
from audio_utils.wav_loader import DEFAULT_AUDIO_PATH, load_wav_cpu
from dali_pipeline.audio_decode import dali_available, decode_audio_with_dali
from visualization.plot_waveform import save_waveform_plot_from_array


def _load_with_backend(input_path: Path, backend: str):
    if backend == "cpu":
        return load_wav_cpu(input_path)

    if backend == "dali":
        if not dali_available():
            raise ImportError("DALI backend was requested, but NVIDIA DALI is not installed.")
        audio_batch, sr_batch = decode_audio_with_dali(input_path)
        audio = np.asarray(audio_batch[0], dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        sr = int(np.asarray(sr_batch).reshape(-1)[0])
        return audio, sr

    raise ValueError(f"Unsupported backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU Audio Visualizer preprocessing entry point.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_AUDIO_PATH), help="Path to input WAV file.")
    parser.add_argument("--backend", choices=["cpu", "dali"], default="cpu", help="Decode backend to use.")
    parser.add_argument("--plot", action="store_true", help="Save a waveform PNG next to the project root.")
    parser.add_argument("--plot-out", type=str, default="waveform_plot.png", help="Output path for waveform plot.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    audio, sample_rate = _load_with_backend(input_path, args.backend)

    print(f"==== {args.backend.upper()} LOAD ====")
    print("Input file:", input_path)
    print("Sample rate:", sample_rate)
    print("Samples:", len(audio))
    print("Duration:", round(len(audio) / sample_rate, 2), "s")

    summary = summarize_features(audio)
    print("Feature summary:")
    print(json.dumps(summary, indent=2))

    if args.plot:
        save_waveform_plot_from_array(audio, sample_rate, args.plot_out)


if __name__ == "__main__":
    main()
