from pathlib import Path
from typing import Tuple
import wave

import numpy as np


def _load_wav_builtin(path):
    with wave.open(str(path), 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        num_frames = wf.getnframes()
        raw = wf.readframes(num_frames)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype='<i4').astype(np.float32) / 2147483648.0
    else:
        raise ValueError('Unsupported WAV sample width: {}'.format(sample_width))

    if num_channels > 1:
        data = data.reshape(-1, num_channels).mean(axis=1)

    return np.ascontiguousarray(data, dtype=np.float32), int(sample_rate)


def load_audio(path):
    path = Path(path)
    if path.suffix.lower() == '.wav':
        return _load_wav_builtin(path)

    try:
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            'This project supports WAV files with no extra packages. '
            'For non-WAV files, install soundfile manually.'
        )

    audio, sample_rate = sf.read(str(path), dtype='float32', always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    return audio, int(sample_rate)
