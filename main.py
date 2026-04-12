from audio_utils.wav_loader import load_wav_cpu
from audio_utils.features import summarize_features

def main():
    input_file = "data/audio/french_ballet_class.wav"

    print("==== CPU LOAD ====")
    cpu_audio, cpu_sr = load_wav_cpu(input_file)
    print("CPU sample rate:", cpu_sr)
    print("CPU samples:", len(cpu_audio))
    print("CPU duration:", round(len(cpu_audio) / cpu_sr, 2), "s")


    cpu_summary = summarize_features(cpu_audio)
    print("CPU feature summary:", cpu_summary)

if name == "main":
    main()