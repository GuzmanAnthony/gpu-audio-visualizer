from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import numpy as np
import os

class AudioDecodePipeline(Pipeline):
    def init(self, file_path, batch_size=1, num_threads=2, device_id=0):
        super().init(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
        self.file_path = file_path

    def definegraph(self):
        encoded,  = fn.readers.file(
            files=[self.file_path],
            labels=[0],
            random_shuffle=False
        )
        audio, sample_rate = fn.decoders.audio(encoded, device="cpu")
        return audio, sample_rate

def decode_audio_with_dali(file_path):
    print("Decoding audio file with DALI:", os.path.abspath(file_path))
    pipe = AudioDecodePipeline(file_path=os.path.abspath(file_path))
    pipe.build()
    audio, sample_rate = pipe.run()
    return audio.as_array(), sample_rate.as_array()

if name == "main":
    audio, sr = decode_audio_with_dali("data/audio/french_ballet_class.wav")
    print("DALI audio decode complete")
    print("Audio shape:", audio.shape)
    print("Sample rate:", sr)
