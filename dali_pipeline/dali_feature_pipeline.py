from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from audio_utils.features import summarize_features
from audio_utils.wav_loader import DEFAULT_AUDIO_PATH
from dali_pipeline.audio_decode import decode_audio_with_dali
from visualization.plot_waveform import save_waveform_plot_from_array


def run_dali_pipeline(
    input_file: str | Path = DEFAULT_AUDIO_PATH,
    output_prefix: str = "dali",
) -> Tuple[np.ndarray, int, dict]:
    print("==== DALI DECODE ====")
    audio_batch, sr_batch = decode_audio_with_dali(input_file)

    audio = np.asarray(audio_batch[0], dtype=np.float32)
    sr = int(np.asarray(sr_batch).reshape(-1)[0])

    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    print(f"Sample rate : {sr} Hz")
    print(f"Samples     : {len(audio)}")
    print(f"Duration    : {len(audio) / sr:.2f}s")
    print(f"Min/Max     : {audio.min():.4f} / {audio.max():.4f}")

    print("\n==== FEATURE SUMMARY ====")
    summary = summarize_features(audio, sample_rate=sr)
    print(summary)

    print("\n==== SAVING WAVEFORM PLOT ====")
    save_waveform_plot_from_array(audio, sr, f"{output_prefix}_waveform.png", title="Waveform of Input Audio (DALI decoded)")

    return audio, sr, summary


if __name__ == "__main__":
    run_dali_pipeline()
