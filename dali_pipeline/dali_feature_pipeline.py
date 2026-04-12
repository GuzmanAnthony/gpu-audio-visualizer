import numpy as np
from dali_pipeline.audio_decode import decode_audio_with_dali
from audio_utils.features import summarize_features
from visualization.plot_waveform import save_waveform_plot_from_array

def run_dali_pipeline(file_root="data/audio", output_prefix="dali"):
    print("==== DALI DECODE ====")
    audio_batch, sr_batch = decode_audio_with_dali("data/audio/french_ballet_class.wav")

    # DALI returns a batch — take the first item
    audio = audio_batch[0]
    sr    = int(sr_batch[0])

    # Flatten to 1D if DALI returned (samples, channels)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    audio = audio.astype(np.float32)

    print(f"Sample rate : {sr} Hz")
    print(f"Samples     : {len(audio)}")
    print(f"Duration    : {len(audio)/sr:.2f}s")
    print(f"Min/Max     : {audio.min():.4f} / {audio.max():.4f}")

    print("\n==== FEATURE SUMMARY ====")
    summary = summarize_features(audio)
    print(summary)

    print("\n==== SAVING WAVEFORM PLOT ====")
    save_waveform_plot_from_array(audio, sr, f"{output_prefix}_waveform.png")

    return audio, sr, summary

if name == "main":
    run_dali_pipeline()