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


def compute_fft_bands(
    signal: np.ndarray,
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> dict:
    """
    Compute per-frame frequency band RMS using FFT.
    Returns bass, mid, treble, and overall level arrays normalized 0-1.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    frames = frame_audio(signal, frame_size=frame_size, hop_size=hop_size)

    if frames.size == 0:
        return {"times": [], "bass": [], "mid": [], "treble": [], "level": [], "sr": sample_rate, "hop": hop_size}

    # Hann window to reduce spectral leakage
    window = np.hanning(frame_size).astype(np.float32)
    windowed = frames * window  # (n_frames, frame_size)

    # FFT — only positive frequencies needed
    fft_mag = np.abs(np.fft.rfft(windowed, axis=1))  # (n_frames, frame_size//2 + 1)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

    # Frequency band masks
    bass_mask   = freqs < 200
    mid_mask    = (freqs >= 200) & (freqs < 2000)
    treble_mask = freqs >= 2000

    def band_rms(mask):
        if not np.any(mask):
            return np.zeros(len(frames), dtype=np.float32)
        return np.sqrt(np.mean(fft_mag[:, mask] ** 2, axis=1)).astype(np.float32)

    def norm(x):
        mx = x.max()
        return (x / (mx + 1e-8)).tolist()

    bass   = band_rms(bass_mask)
    mid    = band_rms(mid_mask)
    treble = band_rms(treble_mask)
    level  = np.sqrt(np.mean(fft_mag ** 2, axis=1)).astype(np.float32)

    times = (np.arange(len(frames)) * hop_size / sample_rate).tolist()

    return {
        "times":  times,
        "bass":   norm(bass),
        "mid":    norm(mid),
        "treble": norm(treble),
        "level":  norm(level),
        "sr":     sample_rate,
        "hop":    hop_size,
    }


def summarize_features(
    signal: np.ndarray,
    sample_rate: int = 44100,      # ← added parameter
    frame_size: int = 1024,
    hop_size: int = 512,
) -> dict:
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

    bands = compute_fft_bands(signal, sample_rate, frame_size=2048, hop_size=hop_size)

    return {
        "num_frames": int(len(frames)),
        "frame_size": int(frame_size),
        "hop_size":   int(hop_size),
        "rms_mean":   float(np.mean(rms)),
        "rms_max":    float(np.max(rms)),
        "peak_mean":  float(np.mean(peak)),
        "peak_max":   float(np.max(peak)),
        "bands":      bands,
    }


if __name__ == "__main__":
    test = np.random.randn(16000).astype(np.float32)
    print(summarize_features(test, sample_rate=44100))