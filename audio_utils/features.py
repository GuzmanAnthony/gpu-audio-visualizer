import numpy as np

def frame_audio(signal, frame_size=1024, hop_size=512):
    frames = []
    for start in range(0, len(signal) - frame_size + 1, hop_size):
        frames.append(signal[start:start + frame_size])
    return np.array(frames, dtype=np.float32)

def compute_rms(frames):
    return np.sqrt(np.mean(frames ** 2, axis=1))

def compute_peak(frames):
    return np.max(np.abs(frames), axis=1)

def summarize_features(signal, frame_size=1024, hop_size=512):
    frames = frame_audio(signal, frame_size, hop_size)
    rms = compute_rms(frames)
    peak = compute_peak(frames)
    return {
        "num_frames": len(frames),
        "rms_mean": float(np.mean(rms)),
        "rms_max": float(np.max(rms)),
        "peak_mean": float(np.mean(peak)),
        "peak_max": float(np.max(peak)),
    }

if __name__ == "__main__":
    test = np.random.randn(16000).astype(np.float32)
    summary = summarize_features(test)
    print(summary)