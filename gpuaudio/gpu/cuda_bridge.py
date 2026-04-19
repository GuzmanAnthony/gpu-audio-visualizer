import ctypes
from pathlib import Path
from typing import Dict, Optional

import numpy as np


class GpuTimings(ctypes.Structure):
    _fields_ = [
        ("h2d_ms", ctypes.c_float),
        ("rms_peak_ms", ctypes.c_float),
        ("window_ms", ctypes.c_float),
        ("fft_ms", ctypes.c_float),
        ("band_centroid_ms", ctypes.c_float),
        ("waveform_ms", ctypes.c_float),
        ("d2h_ms", ctypes.c_float),
    ]


def get_default_cuda_library_path() -> Path:
    return Path(__file__).resolve().parents[2] / "cuda_backend" / "build" / "libgpuaudio_features.so"


def _load_library(path: Optional[str] = None):
    lib_path = Path(path) if path else get_default_cuda_library_path()
    if not lib_path.exists():
        raise FileNotFoundError(f"CUDA shared library not found: {lib_path}")
    lib = ctypes.CDLL(str(lib_path))
    lib.extract_audio_features.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(GpuTimings),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.extract_audio_features.restype = ctypes.c_int
    return lib


def get_cuda_backend_error(path: Optional[str] = None) -> str:
    try:
        _load_library(path)
        return ""
    except Exception as exc:
        return str(exc)


def cuda_backend_available(path: Optional[str] = None) -> bool:
    return get_cuda_backend_error(path) == ""


def _num_frames(num_samples: int, frame_size: int, hop_size: int) -> int:
    if num_samples <= 0:
        return 0
    if num_samples <= frame_size:
        return 1
    return 1 + ((num_samples - frame_size + hop_size - 1) // hop_size)


def compute_feature_bundle_gpu(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int,
    hop_size: int,
    waveform_buckets: int,
    library_path: Optional[str] = None,
) -> Dict:
    audio = np.ascontiguousarray(audio.astype(np.float32, copy=False))
    num_frames = _num_frames(len(audio), frame_size, hop_size)

    rms = np.zeros(num_frames, dtype=np.float32)
    peak = np.zeros(num_frames, dtype=np.float32)
    bass = np.zeros(num_frames, dtype=np.float32)
    mid = np.zeros(num_frames, dtype=np.float32)
    treble = np.zeros(num_frames, dtype=np.float32)
    centroid = np.zeros(num_frames, dtype=np.float32)
    wave_min = np.zeros(waveform_buckets, dtype=np.float32)
    wave_max = np.zeros(waveform_buckets, dtype=np.float32)

    timings = GpuTimings()
    error_buffer = ctypes.create_string_buffer(1024)
    lib = _load_library(library_path)

    rc = lib.extract_audio_features(
        audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(len(audio)),
        ctypes.c_int(sample_rate),
        ctypes.c_int(frame_size),
        ctypes.c_int(hop_size),
        ctypes.c_int(waveform_buckets),
        rms.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        peak.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bass.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        mid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        treble.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        centroid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wave_min.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wave_max.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(timings),
        error_buffer,
        ctypes.c_int(len(error_buffer)),
    )

    if rc != 0:
        raise RuntimeError(f"CUDA feature extraction failed: {error_buffer.value.decode('utf-8', errors='ignore')}")

    metadata = {
        "frame_size": frame_size,
        "hop_size": hop_size,
        "num_frames": int(len(rms)),
        "waveform_buckets": int(len(wave_min)),
        "sample_rate": int(sample_rate),
        "num_samples": int(len(audio)),
        "feature_source": "cuda",
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

    total_gpu_ms = (
        timings.h2d_ms
        + timings.rms_peak_ms
        + timings.window_ms
        + timings.fft_ms
        + timings.band_centroid_ms
        + timings.waveform_ms
        + timings.d2h_ms
    )

    return {
        "metadata": metadata,
        "summary": summary,
        "timings": {
            "feature_wall_ms": float(total_gpu_ms),
            "gpu_h2d_ms": float(timings.h2d_ms),
            "gpu_rms_peak_ms": float(timings.rms_peak_ms),
            "gpu_window_ms": float(timings.window_ms),
            "gpu_fft_ms": float(timings.fft_ms),
            "gpu_band_centroid_ms": float(timings.band_centroid_ms),
            "gpu_waveform_ms": float(timings.waveform_ms),
            "gpu_d2h_ms": float(timings.d2h_ms),
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
