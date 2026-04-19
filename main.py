from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from audio_utils.features import (
    DEFAULT_FRAME_SIZE,
    DEFAULT_HOP_SIZE,
    DEFAULT_WAVEFORM_BUCKETS,
    compute_feature_bundle_cpu,
    feature_bundle_to_json_ready,
    write_feature_bundle_json,
)
from audio_utils.wav_loader import DEFAULT_AUDIO_PATH, load_wav_cpu
from dali_pipeline.audio_decode import dali_available, decode_audio_with_dali
from visualization.plot_waveform import save_waveform_plot_from_array


def _load_with_backend(input_path: Path, backend: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    wall_start = time.perf_counter()

    if backend == "cpu":
        audio, sample_rate = load_wav_cpu(input_path)
        return audio, sample_rate, {"decode_wall_ms": (time.perf_counter() - wall_start) * 1000.0}

    if backend == "dali":
        if not dali_available():
            raise ImportError("DALI backend was requested, but NVIDIA DALI is not installed.")

        audio_batch, sr_batch = decode_audio_with_dali(input_path)
        audio = np.asarray(audio_batch[0], dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1, dtype=np.float32)
        sample_rate = int(np.asarray(sr_batch).reshape(-1)[0])
        return audio, sample_rate, {"decode_wall_ms": (time.perf_counter() - wall_start) * 1000.0}

    raise ValueError(f"Unsupported decode backend: {backend}")


def _compute_features(
    signal: np.ndarray,
    sample_rate: int,
    backend: str,
    frame_size: int,
    hop_size: int,
    waveform_buckets: int,
) -> Dict[str, Any]:
    wall_start = time.perf_counter()

    if backend == "cpu":
        bundle = compute_feature_bundle_cpu(
            signal,
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
            waveform_buckets=waveform_buckets,
        )
        bundle["timings_ms"] = {
            **bundle.get("timings_ms", {}),
            "feature_wall": (time.perf_counter() - wall_start) * 1000.0,
        }
        return bundle

    if backend == "gpu":
        from gpu_features import compute_feature_bundle_gpu

        bundle = compute_feature_bundle_gpu(
            signal,
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
            waveform_buckets=waveform_buckets,
        )
        bundle["timings_ms"] = {
            **bundle.get("timings_ms", {}),
            "feature_wall": (time.perf_counter() - wall_start) * 1000.0,
        }
        return bundle

    raise ValueError(f"Unsupported feature backend: {backend}")


def _print_bundle_summary(input_path: Path, decode_backend: str, feature_backend: str, bundle: Dict[str, Any], decode_timing: Dict[str, Any]) -> None:
    print(f"==== PIPELINE SUMMARY ({decode_backend.upper()} decode -> {feature_backend.upper()} features) ====")
    print("Input file:", input_path)
    print("Sample rate:", bundle["sample_rate"])
    print("Samples:", bundle["num_samples"])
    print("Duration:", round(bundle["duration_seconds"], 2), "s")
    print("Frames:", bundle["num_frames"])
    print("Waveform buckets:", bundle["waveform_buckets"])
    print("Summary:")
    print(json.dumps(bundle["summary"], indent=2))
    print("Timings (ms):")
    print(json.dumps({**decode_timing, **bundle.get("timings_ms", {})}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU audio feature extraction entry point. Decode can stay on CPU or DALI, while feature extraction runs on CPU or CUDA."
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_AUDIO_PATH), help="Path to input WAV file.")
    parser.add_argument("--decode-backend", choices=["cpu", "dali"], default="cpu", help="Decode backend to use.")
    parser.add_argument("--feature-backend", choices=["cpu", "gpu"], default="cpu", help="Feature backend to use.")
    parser.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE, help="Frame size used for RMS, peak, and FFT features.")
    parser.add_argument("--hop-size", type=int, default=DEFAULT_HOP_SIZE, help="Hop size between analysis frames.")
    parser.add_argument("--waveform-buckets", type=int, default=DEFAULT_WAVEFORM_BUCKETS, help="Number of downsampled waveform buckets for the frontend.")
    parser.add_argument("--export-json", type=str, default="", help="Optional output path for a feature JSON file that the browser app can load.")
    parser.add_argument("--plot", action="store_true", help="Save a waveform PNG.")
    parser.add_argument("--plot-out", type=str, default="waveform_plot.png", help="Output path for waveform plot.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    audio, sample_rate, decode_timing = _load_with_backend(input_path, args.decode_backend)
    bundle = _compute_features(
        audio,
        sample_rate=sample_rate,
        backend=args.feature_backend,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        waveform_buckets=args.waveform_buckets,
    )

    _print_bundle_summary(input_path, args.decode_backend, args.feature_backend, bundle, decode_timing)

    if args.export_json:
        output_path = write_feature_bundle_json(
            bundle,
            args.export_json,
            source_audio=input_path.name,
        )
        print(f"Feature JSON written to: {output_path}")

    if args.plot:
        save_waveform_plot_from_array(audio, sample_rate, args.plot_out)
        print(f"Waveform plot written to: {Path(args.plot_out).resolve()}")


if __name__ == "__main__":
    main()
