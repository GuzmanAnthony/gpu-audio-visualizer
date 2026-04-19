# GPU Audio Visualizer
VIST/ECEN 489 – Spring 2026  
Anthony Guzman & Jason Agnew

## Overview
This version shifts the project toward a real CUDA centered pipeline.

The browser is no longer treated as the main DSP engine. Instead, the intended flow is:

1. Decode audio on the host side.
2. Run feature extraction in native CUDA.
3. Export compact feature arrays.
4. Load those CUDA generated features into the browser visualizer.

That makes the browser a renderer and playback surface, while the heavy analysis work happens in CUDA.

## What is new in this update
- Added a native CUDA backend in `cuda_backend/`.
- Added a Python wrapper in `gpu_features.py` that calls the shared library through `ctypes`.
- Reworked feature extraction so the CPU and GPU paths produce the same bundle structure.
- Added CUDA kernels for:
  - fused RMS and peak reduction
  - waveform min and max downsampling
  - Hann windowing
  - spectral band energy and centroid extraction after cuFFT
- Added cuFFT based batched spectral analysis.
- Added timing breakdowns for H2D, kernels, FFT, D2H, GPU only, and full wall time.
- Added a CPU vs GPU benchmark for feature extraction.
- Updated the browser app so it can load exported CUDA feature JSON and use that as the primary animation source.

## Project structure
- `audio_utils/` – CPU WAV loading and CPU reference feature extraction
- `cuda_backend/` – CUDA C++ shared library and build files
- `gpu_features.py` – Python wrapper for the CUDA backend
- `benchmarks/compare_features.py` – CPU vs GPU feature benchmark
- `dali_pipeline/` – optional DALI decode experiments
- `visualization/` – waveform plotting utilities
- `app/` – browser visualizer that can use CUDA exported feature JSON
- `data/audio/` – bundled sample audio

## CUDA backend design
The CUDA backend expects mono float audio and computes:
- frame RMS
- frame peak
- frame bass energy
- frame mid energy
- frame treble energy
- frame spectral centroid
- waveform min and max envelope buckets

### CUDA mapping
- **One block per frame** for RMS and peak reduction
- **Shared memory reductions** for per frame sum of squares and maxima
- **Batched cuFFT** for frame spectra
- **One block per frame** for band energy accumulation and centroid calculation
- **One block per waveform bucket** for min and max downsampling

## Python setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build the CUDA backend
From the repo root:

```bash
cd cuda_backend
./build.sh
cd ..
```

This builds `cuda_backend/build/libgpuaudio_features.so`.

## Run the CPU reference path
```bash
python main.py --input data/audio/french_ballet_class.wav --decode-backend cpu --feature-backend cpu --plot
```

## Run the CUDA feature path
```bash
python main.py \
  --input data/audio/french_ballet_class.wav \
  --decode-backend cpu \
  --feature-backend gpu \
  --export-json outputs/french_ballet_class_features.json
```

## Optional DALI decode experiment
DALI remains an experiment for decode side comparisons. The CUDA work for the class project is in the custom backend, not in DALI.

```bash
python main.py \
  --input data/audio/french_ballet_class.wav \
  --decode-backend dali \
  --feature-backend gpu
```

## Benchmark CPU vs GPU features
```bash
python -m benchmarks.compare_features \
  --input data/audio/french_ballet_class.wav \
  --runs 10 \
  --warmup 2
```

The benchmark reports:
- CPU mean, median, std, min, max
- GPU mean, median, std, min, max
- average GPU kernel timing breakdown
- max absolute difference between CPU and GPU outputs
- overall CPU over GPU speedup

## Browser visualizer
Serve the repo root with a simple static server:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/app/index.html
```

### Browser workflow
1. Load an audio file.
2. Export a feature JSON from the Python CUDA pipeline.
3. Load that JSON in the browser.
4. Press play.

When the JSON is loaded, the visualizer uses CUDA generated RMS, band energy, centroid, and waveform envelope data instead of relying only on the browser AnalyserNode path.

## Notes
- The decode side can still be CPU or DALI depending on the machine.
- The main course aligned work is the custom CUDA feature extraction path.
- The CPU implementation remains in the repo as a correctness baseline and benchmarking reference.
