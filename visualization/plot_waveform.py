from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from audio_utils.wav_loader import load_wav_cpu


def _render_plot(audio: np.ndarray, sample_rate: int, title: str, output_png: str | os.PathLike[str]) -> Path:
    time_axis = np.arange(len(audio), dtype=np.float32) / float(sample_rate)
    output_path = Path(output_png).expanduser().resolve()

    plt.figure(figsize=(12, 4.5))
    plt.plot(time_axis, audio, linewidth=0.6)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def save_waveform_plot(
    input_wav: str | os.PathLike[str] | None = None,
    output_png: str | os.PathLike[str] = "waveform_plot.png",
) -> Path:
    audio, sample_rate = load_wav_cpu(input_wav)
    output_path = _render_plot(audio, sample_rate, "Waveform of Input Audio", output_png)
    print(f"Waveform plot saved to {output_path}")
    return output_path


def save_waveform_plot_from_array(
    audio: np.ndarray,
    sample_rate: int,
    output_png: str | os.PathLike[str] = "waveform_plot.png",
    title: str = "Waveform of Input Audio",
) -> Path:
    output_path = _render_plot(np.asarray(audio, dtype=np.float32), sample_rate, title, output_png)
    print(f"Waveform plot saved to {output_path}")
    return output_path


if __name__ == "__main__":
    save_waveform_plot()
