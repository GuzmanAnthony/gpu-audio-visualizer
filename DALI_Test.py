from dali_pipeline.test_dali_basic import BasicGpuPipe


if __name__ == "__main__":
    p = BasicGpuPipe()
    p.build()
    out = p.run()[0]
    arr = out.as_cpu().as_array()
    print("✅ DALI ran on GPU.")
    print("shape:", arr.shape, "dtype:", arr.dtype, "min/max:", float(arr.min()), float(arr.max()))
