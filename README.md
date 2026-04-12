# GPU Audio Visualizer
VIST/ECEN 489 – Spring 2026  
Anthony Guzman & Jason Agnew

## Overview
This project now has **two integrated parts**:

1. **Python preprocessing + benchmarking tools** for offline audio loading, framing, feature extraction, waveform export, and optional DALI experiments.
2. **A browser-based 3D procedural audio visualizer app** that lets you upload a pre-recorded audio file and render it as a reactive glowing orb.

The app portion was inspired by the uploaded Three.js example, but it has been reworked into this repo so it fits the ECEN/VIST 489 project structure and supports local file upload.

## What Was Polished
- Fixed broken `__name__ == "__main__"` guards.
- Fixed the DALI pipeline class method names.
- Added safer path handling around the bundled sample audio.
- Made feature framing more efficient and robust.
- Cleaned up plotting and CLI behavior.
- Added a proper browser UI for file upload, playback, and live 3D visualization.
- Kept DALI optional so the repo still works even on systems without DALI installed.

## Project Structure
- `audio_utils/` – WAV loading and frame-based feature extraction
- `dali_pipeline/` – DALI decode experiments and simple GPU sanity checks
- `benchmarks/` – CPU vs DALI timing helpers
- `visualization/` – waveform plot export utilities
- `app/` – interactive 3D procedural audio visualizer UI
- `data/audio/` – bundled sample audio

## Python Setup
Create a virtual environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to use DALI, install the package that matches your CUDA/toolkit environment.

## Python Usage
Run the offline preprocessing path:

```bash
python main.py --backend cpu --plot
```

Use the bundled sample explicitly:

```bash
python main.py --input data/audio/french_ballet_class.wav --backend cpu --plot
```

Run the DALI feature pipeline:

```bash
python -m dali_pipeline.dali_feature_pipeline
```

Run the decode benchmark:

```bash
python -m benchmarks.compare_decode --runs 5
```

## 3D Visualizer App
The app is a **static browser app**. You do not need Node to run it.

From the repository root, start a simple local server:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/app/index.html
```

### App Features
- Upload your own prerecorded audio file
- Load the bundled sample audio
- Play / pause audio
- Drag-and-drop audio support
- Bloom, color, and animation controls
- Live bass / mid / treble-driven mesh deformation
- Orbital camera controls

## Notes
- The browser UI uses **Web Audio API + WebGL/Three.js** for real-time visualization.
- The Python side remains useful for offline measurements, plots, and benchmarking.
- The current DALI path is still experimental and is kept separate from the browser UI.
