import argparse
import json
import time
from pathlib import Path

from gpuaudio.io.audio_io import load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional DALI decode benchmark")
    parser.add_argument("--input", required=True)
    parser.add_argument("--runs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    cpu_times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        audio, sample_rate = load_audio(input_path)
        cpu_times.append((time.perf_counter() - start) * 1000.0)

    report = {
        "input": str(input_path.resolve()),
        "cpu_decode_mean_ms": sum(cpu_times) / len(cpu_times),
        "cpu_decode_runs_ms": cpu_times,
    }

    try:
        from nvidia.dali import pipeline_def
        import nvidia.dali.fn as fn
    except ImportError:
        report["dali"] = "DALI not installed. Install requirements-optional-dali.txt to run this experiment."
        print(json.dumps(report, indent=2))
        return

    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def audio_pipeline(file_root, file_list):
        encoded, _ = fn.readers.file(file_root=file_root, file_list=file_list)
        audio, rate = fn.decoders.audio(encoded, dtype=fn.types.FLOAT, downmix=True)
        return audio, rate

    temp_list = input_path.with_suffix('.txt')
    temp_list.write_text(f"{input_path.name}\n")

    dali_times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        pipe = audio_pipeline(file_root=str(input_path.parent), file_list=str(temp_list))
        pipe.build()
        audio, rate = pipe.run()
        _ = audio.as_cpu().as_array(), rate.as_cpu().as_array()
        dali_times.append((time.perf_counter() - start) * 1000.0)

    report["dali_decode_mean_ms"] = sum(dali_times) / len(dali_times)
    report["dali_decode_runs_ms"] = dali_times
    
    if temp_list.exists():
        temp_list.unlink()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
