import argparse
import json
import statistics
import time
from pathlib import Path

from gpuaudio.io.audio_io import load_audio
from gpuaudio.features.cpu_features import compute_feature_bundle_cpu
from gpuaudio.gpu.cuda_bridge import compute_feature_bundle_gpu, cuda_backend_available


def _time_call(fn):
    start = time.perf_counter()
    result = fn()
    wall_ms = (time.perf_counter() - start) * 1000.0
    return result, wall_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CPU and CUDA audio feature extraction")
    parser.add_argument("--input", required=True)
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--waveform-buckets", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cuda-lib", type=str, default="")
    return parser.parse_args()


def summarize(values):
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    args = parse_args()
    audio, sample_rate = load_audio(Path(args.input))

    cpu_times = []
    gpu_times = []
    gpu_kernel_breakdown = []

    for _ in range(args.warmup):
        compute_feature_bundle_cpu(audio, sample_rate, args.frame_size, args.hop_size, args.waveform_buckets)
        if cuda_backend_available(args.cuda_lib or None):
            compute_feature_bundle_gpu(audio, sample_rate, args.frame_size, args.hop_size, args.waveform_buckets, args.cuda_lib or None)

    for _ in range(args.runs):
        _, cpu_wall = _time_call(
            lambda: compute_feature_bundle_cpu(audio, sample_rate, args.frame_size, args.hop_size, args.waveform_buckets)
        )
        cpu_times.append(cpu_wall)

        if cuda_backend_available(args.cuda_lib or None):
            gpu_bundle, gpu_wall = _time_call(
                lambda: compute_feature_bundle_gpu(audio, sample_rate, args.frame_size, args.hop_size, args.waveform_buckets, args.cuda_lib or None)
            )
            gpu_times.append(gpu_wall)
            gpu_kernel_breakdown.append(gpu_bundle["timings"])

    report = {
        "input": str(Path(args.input).resolve()),
        "sample_rate": sample_rate,
        "samples": len(audio),
        "cpu": summarize(cpu_times),
    }

    if gpu_times:
        report["gpu_end_to_end"] = summarize(gpu_times)
        report["speedup_mean"] = report["cpu"]["mean_ms"] / report["gpu_end_to_end"]["mean_ms"]
        report["latest_gpu_kernel_breakdown_ms"] = gpu_kernel_breakdown[-1]
    else:
        report["gpu_end_to_end"] = "CUDA backend unavailable"

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
