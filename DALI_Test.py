from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types

class BasicGpuPipe(Pipeline):
    def __init__(self, batch_size=2, num_threads=2, device_id=0):
        super().__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id)

    def define_graph(self):
        x = fn.random.uniform(device="gpu", shape=(4096,), dtype=types.FLOAT)
        y = x * 2.0
        return y

if __name__ == "__main__":
    p = BasicGpuPipe()
    p.build()
    out = p.run()[0]                 # TensorListGPU
    arr = out.as_cpu().as_array()    # bring to CPU just to print
    print("✅ DALI ran on GPU.")
    print("shape:", arr.shape, "dtype:", arr.dtype, "min/max:", float(arr.min()), float(arr.max()))