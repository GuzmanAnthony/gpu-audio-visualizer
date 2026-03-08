from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import os

class AudioDecodePipeline(Pipeline):
    def __init__(self, file_root="data", batch_size=1, num_threads=2, device_id=0):
        super().__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
        self.file_root = file_root
        self.input = fn.readers.file(file_root=file_root, random_shuffle=False)

    def define_graph(self):
        encoded, _ = self.input
        audio, sample_rate = fn.decoders.audio(encoded, device="cpu")
        return audio, sample_rate

def decode_audio_with_dali(file_root="data"):
    print("Reading audio files from:", os.path.abspath(file_root))
    pipe = AudioDecodePipeline(file_root=file_root)
    pipe.build()
    audio, sample_rate = pipe.run()
    return audio.as_array(), sample_rate.as_array()

if __name__ == "__main__":
    audio, sr = decode_audio_with_dali("data")
    print("DALI audio decode complete")
    print("Audio shape:", audio.shape)
    print("Sample rate:", sr)