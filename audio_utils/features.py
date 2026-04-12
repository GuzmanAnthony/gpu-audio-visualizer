from __future__ import annotations

import numpy as np


def frame_audio(signal: np.ndarray, frame_size: int = 1024, hop_size: int = 512) -> np.ndarray:
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
    return np.sqrt(np.mean(frames ** 2, axis=1, dtype=np.float32), dtype=np.float32)


def compute_peak(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.empty((0,), dtype=np.float32)
    return np.max(np.abs(frames), axis=1)


def summarize_features(signal: np.ndarray, frame_size: int = 1024, hop_size: int = 512) -> dict:
    frames = frame_audio(signal, frame_size=frame_size, hop_size=hop_size)
    rms = compute_rms(frames)
    peak = compute_peak(frames)

    if frames.size == 0:
        return {
            "num_frames": 0,
            "frame_size": frame_size,
            "hop_size": hop_size,
            "rms_mean": 0.0,
            "rms_max": 0.0,
            "peak_mean": 0.0,
            "peak_max": 0.0,
        }

    return {
        "num_frames": int(len(frames)),
        "frame_size": int(frame_size),
        "hop_size": int(hop_size),
        "rms_mean": float(np.mean(rms)),
        "rms_max": float(np.max(rms)),
        "peak_mean": float(np.mean(peak)),
        "peak_max": float(np.max(peak)),
    }


if __name__ == "__main__":
    test = np.random.randn(16000).astype(np.float32)
    print(summarize_features(test))
