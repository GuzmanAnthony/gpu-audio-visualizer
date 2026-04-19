from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


class AudioFeatureResult(ctypes.Structure):
    _fields_ = [
        ("num_frames", ctypes.c_int),
        ("frame_size", ctypes.c_int),
        ("hop_size", ctypes.c_int),
        ("waveform_buckets", ctypes.c_int),
        ("num_samples", ctypes.c_int),
        ("sample_rate", ctypes.c_int),
        ("duration_seconds", ctypes.c_float),
        ("rms", ctypes.POINTER(ctypes.c_float)),
        ("peak", ctypes.POINTER(ctypes.c_float)),
        ("bass", ctypes.POINTER(ctypes.c_float)),
        ("mid", ctypes.POINTER(ctypes.c_float)),
        ("treble", ctypes.POINTER(ctypes.c_float)),
        ("centroid", ctypes.POINTER(ctypes.c_float)),
        ("waveform_min", ctypes.POINTER(ctypes.c_float)),
        ("waveform_max", ctypes.POINTER(ctypes.c_float)),
        ("h2d_ms", ctypes.c_float),
        ("rms_peak_ms", ctypes.c_float),
        ("window_pack_ms", ctypes.c_float),
        ("fft_ms", ctypes.c_float),
        ("band_energy_ms", ctypes.c_float),
        ("waveform_minmax_ms", ctypes.c_float),
        ("d2h_ms", ctypes.c_float),
        ("total_gpu_ms", ctypes.c_float),
        ("total_wall_ms", ctypes.c_float),
        ("error", ctypes.c_char * 512),
    ]


def get_default_cuda_library_path() -> Path:
    env_path = os.environ.get("GPU_AUDIO_CUDA_LIB")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "cuda_backend" / "build" / "libgpuaudio_features.so"


def _load_library() -> ctypes.CDLL:
    lib_path = get_default_cuda_library_path()
    if not lib_path.exists():
        raise FileNotFoundError(
            f"CUDA feature library not found at {lib_path}. Build it first with cuda_backend/build.sh."
        )

    library = ctypes.CDLL(str(lib_path))
    library.compute_audio_features_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(AudioFeatureResult),
    ]
    library.compute_audio_features_cuda.restype = ctypes.c_int
    library.free_audio_feature_result.argtypes = [ctypes.POINTER(AudioFeatureResult)]
    library.free_audio_feature_result.restype = None
    return library


def cuda_backend_available() -> bool:
    try:
        _load_library()
        return True
    except Exception:
        return False


def _copy_ptr_to_array(ptr: ctypes.POINTER(ctypes.c_float), length: int) -> np.ndarray:
    if not bool(ptr) or length <= 0:
        return np.empty((0,), dtype=np.float32)
    return np.ctypeslib.as_array(ptr, shape=(length,)).copy().astype(np.float32, copy=False)


def compute_feature_bundle_gpu(
    signal: np.ndarray,
    sample_rate: int,
    frame_size: int = 1024,
    hop_size: int = 512,
    waveform_buckets: int = 2048,
) -> Dict[str, Any]:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("Signal must contain at least one sample.")

    library = _load_library()
    result = AudioFeatureResult()
    signal_ctypes = np.ascontiguousarray(signal, dtype=np.float32)

    status = library.compute_audio_features_cuda(
        signal_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(signal_ctypes.size),
        int(sample_rate),
        int(frame_size),
        int(hop_size),
        int(waveform_buckets),
        ctypes.byref(result),
    )

    try:
        if status != 0:
            error_text = bytes(result.error).split(b"\0", 1)[0].decode("utf-8", errors="replace")
            raise RuntimeError(f"CUDA feature extraction failed: {error_text or 'unknown error'}")

        num_frames = int(result.num_frames)
        waveform_count = int(result.waveform_buckets)
        frame_times = (
            (np.arange(num_frames, dtype=np.float32) * int(result.hop_size) + 0.5 * int(result.frame_size))
            / float(result.sample_rate)
            if num_frames > 0
            else np.empty((0,), dtype=np.float32)
        )
        waveform_times = (
            (np.arange(waveform_count, dtype=np.float32) + 0.5)
            * (float(result.num_samples) / max(1, waveform_count))
            / float(result.sample_rate)
            if waveform_count > 0
            else np.empty((0,), dtype=np.float32)
        )

        rms = _copy_ptr_to_array(result.rms, num_frames)
        peak = _copy_ptr_to_array(result.peak, num_frames)
        bass = _copy_ptr_to_array(result.bass, num_frames)
        mid = _copy_ptr_to_array(result.mid, num_frames)
        treble = _copy_ptr_to_array(result.treble, num_frames)
        centroid = _copy_ptr_to_array(result.centroid, num_frames)
        waveform_min = _copy_ptr_to_array(result.waveform_min, waveform_count)
        waveform_max = _copy_ptr_to_array(result.waveform_max, waveform_count)

        summary = {
            "num_frames": num_frames,
            "rms_mean": float(np.mean(rms)) if rms.size else 0.0,
            "rms_max": float(np.max(rms)) if rms.size else 0.0,
            "peak_mean": float(np.mean(peak)) if peak.size else 0.0,
            "peak_max": float(np.max(peak)) if peak.size else 0.0,
            "bass_mean": float(np.mean(bass)) if bass.size else 0.0,
            "mid_mean": float(np.mean(mid)) if mid.size else 0.0,
            "treble_mean": float(np.mean(treble)) if treble.size else 0.0,
            "centroid_mean": float(np.mean(centroid)) if centroid.size else 0.0,
            "signal_min": float(np.min(signal_ctypes)),
            "signal_max": float(np.max(signal_ctypes)),
        }

        return {
            "backend": "gpu",
            "sample_rate": int(result.sample_rate),
            "num_samples": int(result.num_samples),
            "duration_seconds": float(result.duration_seconds),
            "frame_size": int(result.frame_size),
            "hop_size": int(result.hop_size),
            "num_frames": num_frames,
            "waveform_buckets": waveform_count,
            "timings_ms": {
                "h2d": float(result.h2d_ms),
                "rms_peak": float(result.rms_peak_ms),
                "window_pack": float(result.window_pack_ms),
                "fft": float(result.fft_ms),
                "band_energy": float(result.band_energy_ms),
                "waveform_minmax": float(result.waveform_minmax_ms),
                "d2h": float(result.d2h_ms),
                "total_gpu": float(result.total_gpu_ms),
                "total_wall": float(result.total_wall_ms),
            },
            "summary": summary,
            "frames": {
                "times": frame_times,
                "rms": rms,
                "peak": peak,
                "bass": bass,
                "mid": mid,
                "treble": treble,
                "centroid": centroid,
            },
            "waveform": {
                "times": waveform_times,
                "min": waveform_min,
                "max": waveform_max,
            },
        }
    finally:
        library.free_audio_feature_result(ctypes.byref(result))
