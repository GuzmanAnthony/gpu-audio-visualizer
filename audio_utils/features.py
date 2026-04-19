from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


DEFAULT_FRAME_SIZE = 1024
DEFAULT_HOP_SIZE = 512
DEFAULT_WAVEFORM_BUCKETS = 2048
BASS_MAX_HZ = 250.0
MID_MAX_HZ = 4000.0


def frame_audio(signal: np.ndarray, frame_size: int = DEFAULT_FRAME_SIZE, hop_size: int = DEFAULT_HOP_SIZE) -> np.ndarray:
    """Split 1D audio into overlapping frames using a stride-based view."""
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive integers")
    if signal.size < frame_size:
        return np.empty((0, frame_size), dtype=np.float32)

    n_frames = 1 + (signal.size - frame_size) // hop_size
    shape = (n_frames, frame_size)
    strides = (signal.strides[0] * hop_size, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    return np.array(frames, dtype=np.float32, copy=True)


def compute_rms(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.empty((0,), dtype=np.float32)
    return np.sqrt(np.mean(frames * frames, axis=1, dtype=np.float32), dtype=np.float32)


def compute_peak(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.empty((0,), dtype=np.float32)
    return np.max(np.abs(frames), axis=1).astype(np.float32, copy=False)


def compute_fft_band_features(frames: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
    if frames.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {
            "bass": empty,
            "mid": empty,
            "treble": empty,
            "centroid": empty,
        }

    frame_size = frames.shape[1]
    window = np.hanning(frame_size).astype(np.float32)
    windowed = frames * window[None, :]
    spectrum = np.fft.rfft(windowed, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / float(sample_rate)).astype(np.float32)

    bass_mask = freqs <= BASS_MAX_HZ
    mid_mask = (freqs > BASS_MAX_HZ) & (freqs <= MID_MAX_HZ)
    treble_mask = freqs > MID_MAX_HZ

    bass = power[:, bass_mask].mean(axis=1, dtype=np.float32) if np.any(bass_mask) else np.zeros((power.shape[0],), dtype=np.float32)
    mid = power[:, mid_mask].mean(axis=1, dtype=np.float32) if np.any(mid_mask) else np.zeros((power.shape[0],), dtype=np.float32)
    treble = power[:, treble_mask].mean(axis=1, dtype=np.float32) if np.any(treble_mask) else np.zeros((power.shape[0],), dtype=np.float32)

    weighted_sum = np.sum(power * freqs[None, :], axis=1, dtype=np.float64)
    power_sum = np.sum(power, axis=1, dtype=np.float64)
    centroid = np.divide(
        weighted_sum,
        power_sum + 1e-12,
        out=np.zeros_like(weighted_sum, dtype=np.float64),
    ).astype(np.float32)

    return {
        "bass": bass.astype(np.float32, copy=False),
        "mid": mid.astype(np.float32, copy=False),
        "treble": treble.astype(np.float32, copy=False),
        "centroid": centroid,
    }


def compute_waveform_envelope(signal: np.ndarray, buckets: int = DEFAULT_WAVEFORM_BUCKETS) -> dict[str, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    buckets = max(1, int(buckets))
    if signal.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {"min": empty, "max": empty, "times": empty}

    bucket_edges = np.linspace(0, signal.size, buckets + 1, dtype=np.int64)
    mins = np.empty((buckets,), dtype=np.float32)
    maxs = np.empty((buckets,), dtype=np.float32)
    centers = np.empty((buckets,), dtype=np.float32)

    for bucket_idx in range(buckets):
        start = int(bucket_edges[bucket_idx])
        end = int(bucket_edges[bucket_idx + 1])
        if end <= start:
            end = min(signal.size, start + 1)
        chunk = signal[start:end]
        mins[bucket_idx] = float(np.min(chunk))
        maxs[bucket_idx] = float(np.max(chunk))
        centers[bucket_idx] = 0.5 * (start + end)

    return {
        "min": mins,
        "max": maxs,
        "times": centers,
    }


def _build_summary(
    signal: np.ndarray,
    rms: np.ndarray,
    peak: np.ndarray,
    bass: np.ndarray,
    mid: np.ndarray,
    treble: np.ndarray,
    centroid: np.ndarray,
) -> dict[str, float | int]:
    if signal.size == 0 or rms.size == 0:
        return {
            "num_frames": 0,
            "rms_mean": 0.0,
            "rms_max": 0.0,
            "peak_mean": 0.0,
            "peak_max": 0.0,
            "bass_mean": 0.0,
            "mid_mean": 0.0,
            "treble_mean": 0.0,
            "centroid_mean": 0.0,
            "signal_min": 0.0,
            "signal_max": 0.0,
        }

    return {
        "num_frames": int(rms.size),
        "rms_mean": float(np.mean(rms)),
        "rms_max": float(np.max(rms)),
        "peak_mean": float(np.mean(peak)),
        "peak_max": float(np.max(peak)),
        "bass_mean": float(np.mean(bass)),
        "mid_mean": float(np.mean(mid)),
        "treble_mean": float(np.mean(treble)),
        "centroid_mean": float(np.mean(centroid)),
        "signal_min": float(np.min(signal)),
        "signal_max": float(np.max(signal)),
    }


def compute_feature_bundle_cpu(
    signal: np.ndarray,
    sample_rate: int,
    frame_size: int = DEFAULT_FRAME_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    waveform_buckets: int = DEFAULT_WAVEFORM_BUCKETS,
) -> Dict[str, Any]:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    frames = frame_audio(signal, frame_size=frame_size, hop_size=hop_size)
    rms = compute_rms(frames)
    peak = compute_peak(frames)
    spectral = compute_fft_band_features(frames, sample_rate=sample_rate)
    waveform = compute_waveform_envelope(signal, buckets=waveform_buckets)

    frame_times = (
        (np.arange(rms.size, dtype=np.float32) * hop_size + 0.5 * frame_size) / float(sample_rate)
        if rms.size
        else np.empty((0,), dtype=np.float32)
    )
    waveform_times = waveform["times"] / float(sample_rate) if waveform["times"].size else waveform["times"]

    summary = _build_summary(
        signal,
        rms,
        peak,
        spectral["bass"],
        spectral["mid"],
        spectral["treble"],
        spectral["centroid"],
    )

    return {
        "backend": "cpu",
        "sample_rate": int(sample_rate),
        "num_samples": int(signal.size),
        "duration_seconds": float(signal.size / float(sample_rate)) if sample_rate else 0.0,
        "frame_size": int(frame_size),
        "hop_size": int(hop_size),
        "num_frames": int(rms.size),
        "waveform_buckets": int(waveform_buckets),
        "timings_ms": {},
        "summary": summary,
        "frames": {
            "times": frame_times.astype(np.float32, copy=False),
            "rms": rms.astype(np.float32, copy=False),
            "peak": peak.astype(np.float32, copy=False),
            "bass": spectral["bass"].astype(np.float32, copy=False),
            "mid": spectral["mid"].astype(np.float32, copy=False),
            "treble": spectral["treble"].astype(np.float32, copy=False),
            "centroid": spectral["centroid"].astype(np.float32, copy=False),
        },
        "waveform": {
            "times": waveform_times.astype(np.float32, copy=False),
            "min": waveform["min"].astype(np.float32, copy=False),
            "max": waveform["max"].astype(np.float32, copy=False),
        },
    }


def summarize_features(signal: np.ndarray, frame_size: int = DEFAULT_FRAME_SIZE, hop_size: int = DEFAULT_HOP_SIZE) -> dict:
    bundle = compute_feature_bundle_cpu(signal, sample_rate=1, frame_size=frame_size, hop_size=hop_size, waveform_buckets=1)
    summary = dict(bundle["summary"])
    summary["frame_size"] = int(frame_size)
    summary["hop_size"] = int(hop_size)
    return summary


def _rounded_list(values: np.ndarray, digits: int = 6) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return []
    return np.round(arr, digits).astype(np.float32).tolist()


def feature_bundle_to_json_ready(bundle: Dict[str, Any], source_audio: str | None = None) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "source_audio": source_audio,
        "backend": bundle["backend"],
        "sample_rate": int(bundle["sample_rate"]),
        "num_samples": int(bundle["num_samples"]),
        "duration_seconds": float(bundle["duration_seconds"]),
        "frame_size": int(bundle["frame_size"]),
        "hop_size": int(bundle["hop_size"]),
        "num_frames": int(bundle["num_frames"]),
        "waveform_buckets": int(bundle["waveform_buckets"]),
        "timings_ms": {k: float(v) for k, v in bundle.get("timings_ms", {}).items()},
        "summary": {
            k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
            for k, v in bundle["summary"].items()
        },
        "frames": {
            "times": _rounded_list(bundle["frames"]["times"]),
            "rms": _rounded_list(bundle["frames"]["rms"]),
            "peak": _rounded_list(bundle["frames"]["peak"]),
            "bass": _rounded_list(bundle["frames"]["bass"]),
            "mid": _rounded_list(bundle["frames"]["mid"]),
            "treble": _rounded_list(bundle["frames"]["treble"]),
            "centroid": _rounded_list(bundle["frames"]["centroid"]),
        },
        "waveform": {
            "times": _rounded_list(bundle["waveform"]["times"]),
            "min": _rounded_list(bundle["waveform"]["min"]),
            "max": _rounded_list(bundle["waveform"]["max"]),
        },
    }


def write_feature_bundle_json(bundle: Dict[str, Any], output_path: str | Path, source_audio: str | None = None) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_ready = feature_bundle_to_json_ready(bundle, source_audio=source_audio)
    output_path.write_text(json.dumps(json_ready, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    test = np.random.randn(16000).astype(np.float32)
    result = compute_feature_bundle_cpu(test, sample_rate=16000)
    print(json.dumps(feature_bundle_to_json_ready(result, source_audio="synthetic.wav")["summary"], indent=2))
