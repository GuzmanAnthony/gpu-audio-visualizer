# GPU Audio Visualizer
VIST/ECEN 489 – Spring 2026  
Anthony Guzman & Jason Agnew

## Overview
This project has **two integrated parts**:

1. **Python preprocessing + benchmarking tools** for offline audio loading, framing, feature extraction, waveform export, and DALI GPU-accelerated decoding.
2. **A browser-based 3D procedural audio visualizer** that uploads a recorded audio file, processes it through the Python backend, and renders it as a reactive glowing orb driven by precomputed frequency features.

The visualizer is backed by a **FastAPI server** that handles audio decoding (via DALI or CPU fallback) and FFT-based feature extraction. The browser receives per-frame bass, mid, treble, and level arrays and uses them to drive mesh deformation, bloom, and lighting in real time.

## What Was Polished
- Fixed broken `__name__ == "__main__"` guards.
- Fixed the DALI pipeline class method names.
- Added safer path handling around the bundled sample audio.
- Made feature framing more efficient and robust.
- Cleaned up plotting and CLI behavior.
- Added a proper browser UI for file upload, playback, and live 3D visualization.
- Replaced browser-side Web Audio API FFT with precomputed Python FFT band features (bass, mid, treble, level) served over a `/features` POST endpoint.
- Added multi-format audio upload support with automatic WAV conversion.
- Kept DALI optional so the repo still works on systems without a GPU.

## Project Structure
- `audio_utils/` – WAV loading, frame-based feature extraction, and FFT band computation
- `dali_pipeline/` – DALI GPU decode pipeline and sanity checks
- `benchmarks/` – CPU vs DALI decode and full pipeline timing comparison
- `visualization/` – waveform plot export utilities
- `app/` – interactive 3D audio visualizer (Three.js + WebGL)
- `data/audio/` – bundled sample audio
- `main.py` – FastAPI server + CLI preprocessing entry point

## Python Setup
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Additional dependencies needed for the FastAPI server:

```bash
pip install fastapi uvicorn python-multipart soundfile
```

If you want to use DALI, install the package matching your CUDA version:

```bash
pip install nvidia-dali-cuda120   # for CUDA 12.x
pip install nvidia-dali-cuda110   # for CUDA 11.x
```

## Running the Visualizer (FastAPI Server)

The visualizer requires the FastAPI server to be running. The browser uploads the audio
file to `/features`, which decodes it and returns precomputed FFT band data to drive the orb.

### Local Machine

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open:

```
http://localhost:8000/app/index.html
```

### On HPRC (Texas A&M Grace Cluster)

**Step 1 — Request a GPU node:**

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:1 --time=02:00:00 --pty bash
```

**Step 2 — Note your compute node hostname:**

```bash
hostname   # e.g. gn001
```

**Step 3 — Start the server on the compute node:**

```bash
cd /path/to/gpu-audio-visualizer
conda activate gpu-audio
uvicorn main:app --host 0.0.0.0 --port 6006
```

**Step 4 — On your local machine, open an SSH tunnel:**

```bash
ssh -L 6006:localhost:6006 yournetid@grace.hprc.tamu.edu
```

**Step 5 — Open your local browser:**

```
http://localhost:6006/app/index.html
```

**Tip:** Use `tmux` on the compute node so the server keeps running if your SSH session drops:

```bash
tmux new -s visualizer
uvicorn main:app --host 0.0.0.0 --port 6006
# Detach: Ctrl+B then D
# Reattach: tmux attach -t visualizer
```

## Python CLI Usage

Run offline preprocessing (CPU):

```bash
python main.py --backend cpu --plot
```

Run with DALI decode:

```bash
python main.py --backend dali --input data/audio/french_ballet_class.wav
```

Run the DALI feature pipeline:

```bash
python -m dali_pipeline.dali_feature_pipeline
```

Run the decode + full pipeline benchmark:

```bash
python -m benchmarks.compare_decode --runs 5
```

Save benchmark results for reporting:

```bash
python -m benchmarks.compare_decode --runs 10 | tee benchmark_results.txt
```

## How the Audio Pipeline Works

```
Browser: upload WAV file
      ↓
FastAPI /features endpoint
      ↓
DALI decode (GPU) → falls back to CPU if unavailable
      ↓
FFT band extraction (bass 0–200Hz, mid 200–2000Hz, treble 2000Hz+)
      ↓
Per-frame RMS normalized 0–1, returned as JSON
      ↓
Browser: drives orb uniforms (uBass, uMid, uTreble, uLevel)
         by timestamp lookup during playback
```

## Visualizer Features
- Upload your own WAV audio file
- Load the bundled sample audio
- Play / pause playback
- Drag-and-drop audio support
- Orb deformation driven by precomputed bass, mid, treble, and level
- Bloom, color, wire opacity, and animation controls via GUI panel
- Orbital camera controls (drag to rotate, scroll to zoom)

## GPU Verification

To confirm DALI is using your GPU:

```bash
python -m dali_pipeline.audio_decode
```

Monitor GPU utilization while benchmarking:

```bash
watch -n 0.5 nvidia-smi
```

## Notes
- The browser uses **Three.js + WebGL** for 3D rendering and **Web Audio API** as a live fallback if the server is unreachable.
- The Python backend uses **DALI for GPU-accelerated audio decoding** and **NumPy FFT for frequency band extraction**.
- The DALI decode step runs on CPU (DALI limitation for audio), but transfers data to GPU for downstream processing.
- A custom CUDA kernel for FFT band extraction is a planned milestone to move the full feature pipeline to the GPU.