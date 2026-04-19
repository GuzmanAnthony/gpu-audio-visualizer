import math
import time
from typing import Dict

import numpy as np


BASS_MAX_HZ = 250.0
MID_MAX_HZ = 2000.0
TREBLE_MAX_HZ = 8000.0


def _num_frames(num_samples: int, frame_size: int, hop_size: int) -> int:
    if num_samples <= 0:
        return 0
    if num_samples <= frame_size:
        return 1
    return 1 + math.ceil((num_samples - frame_size) / hop_size)


def _frame_audio(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    count = _num_frames(len(audio), frame_size, hop_size)
    total_needed = frame_size + max(0, count - 1) * hop_size
    padded = np.pad(audio, (0, max(0, total_needed - len(audio))))
    shape = (count, frame_size)
    strides = (padded.strides[0] * hop_size, padded.strides[0])
    return np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides, writeable=False).copy()


def _waveform_min_max(audio: np.ndarray, buckets: int):
    buckets = max(1, buckets)
    bucket_len = math.ceil(len(audio) / buckets)
    total = bucket_len * buckets
    padded = np.pad(audio, (0, total - len(audio)), constant_values=0.0)
    reshaped = padded.reshape(buckets, bucket_len)
    return reshaped.min(axis=1).astype(np.float32), reshaped.max(axis=1).astype(np.float32)


def compute_feature_bundle_cpu(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int,
    hop_size: int,
    waveform_buckets: int,
) -> Dict:
    start = time.perf_counter()
    frames = _frame_audio(audio, frame_size, hop_size)
    window = np.hanning(frame_size).astype(np.float32)

    rms = np.sqrt(np.mean(frames * frames, axis=1, dtype=np.float64)).astype(np.float32)
    peak = np.max(np.abs(frames), axis=1).astype(np.float32)

    fft_in = frames * window[None, :]
    spectrum = np.fft.rfft(fft_in, axis=1)
    mag2 = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate).astype(np.float32)

    bass_mask = freqs < BASS_MAX_HZ
    mid_mask = (freqs >= BASS_MAX_HZ) & (freqs < MID_MAX_HZ)
    treble_mask = (freqs >= MID_MAX_HZ) & (freqs < TREBLE_MAX_HZ)

    bass = mag2[:, bass_mask].sum(axis=1, dtype=np.float64).astype(np.float32)
    mid = mag2[:, mid_mask].sum(axis=1, dtype=np.float64).astype(np.float32)
    treble = mag2[:, treble_mask].sum(axis=1, dtype=np.float64).astype(np.float32)

    weighted = (mag2 * freqs[None, :]).sum(axis=1, dtype=np.float64)
    total = mag2.sum(axis=1, dtype=np.float64) + 1e-12
    centroid = (weighted / total).astype(np.float32)

    wave_min, wave_max = _waveform_min_max(audio, waveform_buckets)

    feature_wall = (time.perf_counter() - start) * 1000.0

    metadata = {
        "frame_size": frame_size,
        "hop_size": hop_size,
        "num_frames": int(len(rms)),
        "waveform_buckets": int(len(wave_min)),
        "sample_rate": int(sample_rate),
        "num_samples": int(len(audio)),
        "feature_source": "cpu",
    }

    summary = {
        "num_frames": int(len(rms)),
        "rms_mean": float(rms.mean()) if len(rms) else 0.0,
        "rms_max": float(rms.max()) if len(rms) else 0.0,
        "peak_mean": float(peak.mean()) if len(peak) else 0.0,
        "peak_max": float(peak.max()) if len(peak) else 0.0,
        "bass_mean": float(bass.mean()) if len(bass) else 0.0,
        "mid_mean": float(mid.mean()) if len(mid) else 0.0,
        "treble_mean": float(treble.mean()) if len(treble) else 0.0,
        "centroid_mean": float(centroid.mean()) if len(centroid) else 0.0,
        "signal_min": float(audio.min()) if len(audio) else 0.0,
        "signal_max": float(audio.max()) if len(audio) else 0.0,
    }

    return {
        "metadata": metadata,
        "summary": summary,
        "timings": {
            "feature_wall_ms": feature_wall,
            "gpu_h2d_ms": 0.0,
            "gpu_rms_peak_ms": 0.0,
            "gpu_window_ms": 0.0,
            "gpu_fft_ms": 0.0,
            "gpu_band_centroid_ms": 0.0,
            "gpu_waveform_ms": 0.0,
            "gpu_d2h_ms": 0.0,
        },
        "features": {
            "rms": rms.tolist(),
            "peak": peak.tolist(),
            "bass": bass.tolist(),
            "mid": mid.tolist(),
            "treble": treble.tolist(),
            "centroid": centroid.tolist(),
            "wave_min": wave_min.tolist(),
            "wave_max": wave_max.tolist(),
        },
    }
