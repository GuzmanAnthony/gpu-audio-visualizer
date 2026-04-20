# GPU Audio Visualizer
VIST/ECEN 489 – Spring 2026  
Anthony Guzman & Jason Agnew

## Overview
This final integrated version keeps the real computation in the backend and uses the browser as the renderer.

The intended pipeline is:

1. Load an offline audio file.
2. Decode on the backend with DALI when available or CPU WAV fallback.
3. Run feature extraction with the native CUDA backend.
4. Return compact feature arrays through the FastAPI server.
5. Render the orb in the browser from backend produced data.

That keeps the project aligned with the GPU class because the feature extraction stage is the main computation and it happens in CUDA rather than in browser DSP code.

## What is in the final integrated version
- Native CUDA backend in `cuda_backend/`
- Python `ctypes` bridge in `gpu_features/cuda_bridge.py`
- FastAPI server in `server.py`
- Frontend orb visualizer in `app/`
- Backend sample and upload routes that feed the orb directly
- Optional DALI decode path for backend audio loading
- CPU reference path kept for debugging and validation

## Project structure
- `app/` – browser orb visualizer
- `audio_utils/` – CPU WAV loading and CPU reference feature extraction
- `cuda_backend/` – CUDA C++ shared library and build script
- `dali_pipeline/` – optional DALI decode helper
- `gpu_features/` – Python bridge into the CUDA shared library
- `benchmarks/` – comparison scripts
- `data/audio/` – bundled sample audio
- `server.py` – integrated backend plus static frontend server

## CUDA feature pipeline
The CUDA backend computes:
- frame RMS
- frame peak
- frame bass energy
- frame mid energy
- frame treble energy
- frame spectral centroid
- waveform min and max display buckets

### CUDA mapping
- one block per frame for RMS and peak reduction
- shared memory reductions inside each block
- batched cuFFT for windowed spectra
- one block per frame for band energy and centroid accumulation
- one block per waveform bucket for min and max downsampling

## Python environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional DALI install
Install DALI only on a machine that has a matching CUDA setup.

```bash
pip install -r requirements-optional-dali.txt
```

## Build the CUDA backend
From the repo root:

```bash
cd cuda_backend
./build.sh
cd ..
```

This should produce:

```text
cuda_backend/build/libgpuaudio_features.so
```

## Run the integrated backend plus frontend
From the repo root:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open the app in a browser:

```text
http://localhost:8000/
```

## Browser workflow
### Bundled sample
1. Click **Load bundled sample from backend**.
2. The audio loads in the browser for playback.
3. The frontend requests backend CUDA features from `/api/sample/features`.
4. The orb renders from backend produced arrays.

### Uploaded audio
1. Upload an audio file.
2. The browser loads it for playback.
3. The same file is uploaded to `/api/features`.
4. The backend decodes the file and runs CUDA feature extraction.
5. The orb renders from the returned feature bundle.

### Manual JSON override
You can still load a pre exported feature JSON manually for offline demos.

## API routes
- `GET /api/health` – backend status, sample list, CUDA availability, DALI availability
- `GET /api/sample/features` – compute features for a bundled sample
- `POST /api/features` – upload audio and compute features

## HPRC usage
On the cluster:

```bash
module load CUDA/11.8.0
source .venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000
```

Then tunnel the port from your local machine and open the app through that forwarded address.

## CPU reference and benchmarking
### CPU reference path
```bash
python main.py --input data/audio/french_ballet_class.wav --decode-backend cpu --feature-backend cpu --plot
```

### GPU feature path
```bash
python main.py \
  --input data/audio/french_ballet_class.wav \
  --decode-backend cpu \
  --feature-backend gpu \
  --export-json outputs/french_ballet_class_features.json
```

### CPU vs GPU benchmark
```bash
python -m benchmarks.compare_features \
  --input data/audio/french_ballet_class.wav \
  --runs 10 \
  --warmup 2
```

## Notes
- DALI is optional and mainly used for backend decode experiments.
- The class critical computation is the custom CUDA feature extraction path.
- The frontend keeps the browser analyser only as a fallback when backend CUDA features are unavailable.
