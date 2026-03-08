import time
import numpy as np
from audio_utils.wav_loader import load_wav_cpu
from dali_pipeline.audio_decode import decode_audio_with_dali

def benchmark_cpu(filepath="data/test.wav", runs=5):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        audio, sr = load_wav_cpu(filepath)
        end = time.perf_counter()
        times.append(end - start)
    return audio, sr, np.array(times)

def benchmark_dali(file_root="data", runs=5):
    times = []
    last_audio = None
    last_sr = None
    for _ in range(runs):
        start = time.perf_counter()
        audio, sr = decode_audio_with_dali(file_root)
        end = time.perf_counter()
        times.append(end - start)
        last_audio = audio
        last_sr = sr
    return last_audio, last_sr, np.array(times)

if __name__ == "__main__":
    cpu_audio, cpu_sr, cpu_times = benchmark_cpu()
    dali_audio, dali_sr, dali_times = benchmark_dali()

    print("\n===== Decode Benchmark Results =====")
    print(f"CPU average decode time : {cpu_times.mean():.6f} s")
    print(f"DALI average decode time: {dali_times.mean():.6f} s")
    print(f"CPU sample rate         : {cpu_sr}")
    print(f"DALI sample rate        : {dali_sr}")
    print(f"CPU samples loaded      : {len(cpu_audio)}")
    print(f"DALI output shape       : {dali_audio.shape}")