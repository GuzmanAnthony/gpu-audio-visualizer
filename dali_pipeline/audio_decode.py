from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
except ImportError:  # pragma: no cover - depends on environment
    Pipeline = None
    fn = None


def dali_available() -> bool:
    return Pipeline is not None and fn is not None


if Pipeline is not None:
    class AudioDecodePipeline(Pipeline):
        def __init__(self, file_path: str | os.PathLike[str], batch_size: int = 1, num_threads: int = 2, device_id: int = 0):
            super().__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
            self.file_path = str(Path(file_path).expanduser().resolve())

        def define_graph(self):
            encoded, _ = fn.readers.file(files=[self.file_path], labels=[0], random_shuffle=False)
            audio, sample_rate = fn.decoders.audio(encoded, device="cpu")
            return audio, sample_rate
else:  # pragma: no cover - only used when DALI is unavailable
    AudioDecodePipeline = None


def decode_audio_with_dali(file_path: str | os.PathLike[str]) -> Tuple[np.ndarray, np.ndarray]:
    if not dali_available():
        raise ImportError(
            "NVIDIA DALI is not installed or not available in this environment. "
            "Install the matching nvidia-dali package for your CUDA version first."
        )

    resolved_path = Path(file_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved_path}")

    print("Decoding audio file with DALI:", resolved_path)
    pipe = AudioDecodePipeline(file_path=resolved_path)
    pipe.build()
    audio, sample_rate = pipe.run()

    audio_np = audio.as_array()
    sr_np = sample_rate.as_array()
    return np.asarray(audio_np), np.asarray(sr_np)


if __name__ == "__main__":
    audio, sr = decode_audio_with_dali(Path(__file__).resolve().parents[1] / "data" / "audio" / "french_ballet_class.wav")
    print("DALI audio decode complete")
    print("Audio shape:", audio.shape)
    print("Sample rate:", sr)
