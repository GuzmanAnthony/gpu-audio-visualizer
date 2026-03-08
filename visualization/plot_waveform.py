import os
import matplotlib.pyplot as plt
from audio_utils.wav_loader import load_wav_cpu

def save_waveform_plot(input_wav="data/audio/french_ballet_class.wav", output_png="waveform_plot.png"):
    audio, sample_rate = load_wav_cpu(input_wav)

    time_axis = [i / sample_rate for i in range(len(audio))]

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, audio, linewidth=0.8)
    plt.title("Waveform of Input Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

    print(f"Waveform plot saved to {os.path.abspath(output_png)}")

if __name__ == "__main__":
    save_waveform_plot()