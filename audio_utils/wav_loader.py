from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

DEFAULT_AUDIO_PATH = Path(__file__).resolve().parents[1] / "data" / "audio" / "french_ballet_class.wav"


def resolve_audio_path(filepath: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a user-provided audio path or fall back to the bundled sample."""
    if filepath is None:
        return DEFAULT_AUDIO_PATH
    return Path(filepath).expanduser().resolve()


def decode_24bit_to_float32(raw_bytes: bytes, n_channels: int) -> np.ndarray:
    """Convert packed 24-bit PCM samples into normalized float32 mono audio."""
    bytes_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    samples = bytes_array.reshape(-1, 3)

    values = (
        samples[:, 0].astype(np.int32)
        | (samples[:, 1].astype(np.int32) << 8)
        | (samples[:, 2].astype(np.int32) << 16)
    )

    sign_mask = 1 << 23
    values = (values ^ sign_mask) - sign_mask

    audio = values.astype(np.float32) / 8388608.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
        audio = np.mean(audio, axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _downmix_if_needed(audio: np.ndarray, n_channels: int) -> np.ndarray:
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
        audio = np.mean(audio, axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_wav_cpu(filepath: str | os.PathLike[str] | None = None) -> Tuple[np.ndarray, int]:
    """Load a WAV file into normalized float32 mono samples.

    Supported sample widths: 8-bit, 16-bit, 24-bit, and 32-bit PCM.
    """
    path = resolve_audio_path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 1:
        audio = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
        audio = _downmix_if_needed(audio, n_channels)
    elif sampwidth == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio = _downmix_if_needed(audio, n_channels)
    elif sampwidth == 3:
        audio = decode_24bit_to_float32(raw_bytes, n_channels)
    elif sampwidth == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        audio = _downmix_if_needed(audio, n_channels)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False), sample_rate


if __name__ == "__main__":
    audio, sr = load_wav_cpu()
    print("CPU WAV load complete")
    print("File:", resolve_audio_path())
    print("Samples:", len(audio))
    print("Sample rate:", sr)
    print("Min/Max:", float(audio.min()), float(audio.max()))
