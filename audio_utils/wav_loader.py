import wave
import numpy as np
import os

def decode_24bit_to_float32(raw_bytes, n_channels):
    bytes_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    samples = bytes_array.reshape(-1, 3)

    # Combine 3 bytes into signed 24-bit integer
    values = (
        samples[:, 0].astype(np.int32) |
        (samples[:, 1].astype(np.int32) << 8) |
        (samples[:, 2].astype(np.int32) << 16)
    )

    # Sign extension for 24-bit
    sign_mask = 1 << 23
    values = (values ^ sign_mask) - sign_mask

    audio = values.astype(np.float32) / 8388608.0  # 2^23
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
        audio = np.mean(audio, axis=1)

    return audio

def load_wav_cpu(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    with wave.open(filepath, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()

        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 1:
        audio = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            audio = np.mean(audio, axis=1)

    elif sampwidth == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            audio = np.mean(audio, axis=1)

    elif sampwidth == 3:
        audio = decode_24bit_to_float32(raw_bytes, n_channels)

    elif sampwidth == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32)
        audio = audio / 2147483648.0

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            audio = np.mean(audio, axis=1)

    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    return audio, sample_rate

if __name__ == "__main__":
    test_file = "data/audio/french_ballet_class.wav"
    audio, sr = load_wav_cpu(test_file)
    print("CPU WAV load complete")
    print("File:", test_file)
    print("Samples:", len(audio))
    print("Sample rate:", sr)
    print("Min/Max:", float(audio.min()), float(audio.max()))