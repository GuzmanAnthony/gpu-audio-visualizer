from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from audio_utils.features import compute_feature_bundle_cpu, feature_bundle_to_json_ready
from audio_utils.wav_loader import load_wav_cpu
from dali_pipeline.audio_decode import dali_available, decode_audio_with_dali
from gpu_features.cuda_bridge import (
    compute_feature_bundle_gpu,
    cuda_backend_available,
    get_cuda_backend_error,
    get_default_cuda_library_path,
)

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / 'app'
DATA_DIR = ROOT / 'data'
SAMPLE_AUDIO_DIR = DATA_DIR / 'audio'

app = FastAPI(title='GPU Audio Visualizer API', version='1.0.0')


def _sample_lookup() -> dict[str, Path]:
    return {path.name: path for path in SAMPLE_AUDIO_DIR.glob('*.wav')}


def _extract_sample_rate(sample_rate_raw: Any) -> int:
    arr = np.asarray(sample_rate_raw)
    if arr.size == 0:
        raise ValueError('Decoded sample rate array was empty.')
    return int(np.ravel(arr)[0])


def _prepare_mono_signal(audio_raw: Any) -> np.ndarray:
    arr = np.asarray(audio_raw, dtype=np.float32)
    if arr.size == 0:
        raise ValueError('Decoded audio was empty.')

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        if arr.shape[1] <= 8:
            arr = arr.mean(axis=1)
        elif arr.shape[0] <= 8:
            arr = arr.mean(axis=0)
        else:
            arr = arr.reshape(-1)
    elif arr.ndim != 1:
        arr = arr.reshape(-1)

    arr = np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)
    if arr.size == 0:
        raise ValueError('Prepared mono signal was empty.')
    return arr


def _decode_audio_file(file_path: Path, decode_backend: str) -> tuple[np.ndarray, int, str]:
    resolved = file_path.resolve()
    requested = decode_backend.lower()
    use_dali = requested in {'auto', 'dali'}

    if use_dali:
        if not dali_available():
            if requested == 'dali':
                raise RuntimeError('DALI decode was requested but NVIDIA DALI is not available in this environment.')
        else:
            try:
                audio_raw, sample_rate_raw = decode_audio_with_dali(resolved)
                return _prepare_mono_signal(audio_raw), _extract_sample_rate(sample_rate_raw), 'dali'
            except Exception as exc:
                if requested == 'dali':
                    raise RuntimeError(f'DALI decode failed: {exc}') from exc

    if resolved.suffix.lower() != '.wav':
        raise RuntimeError('CPU fallback currently supports WAV only. Install DALI and use decode_backend=auto for additional formats.')

    signal, sample_rate = load_wav_cpu(resolved)
    return np.ascontiguousarray(signal, dtype=np.float32), int(sample_rate), 'cpu'


def _compute_feature_bundle(signal: np.ndarray, sample_rate: int, feature_backend: str) -> dict[str, Any]:
    requested = feature_backend.lower()
    if requested == 'gpu':
        if not cuda_backend_available():
            raise RuntimeError(get_cuda_backend_error())
        return compute_feature_bundle_gpu(signal, sample_rate)
    if requested == 'cpu':
        return compute_feature_bundle_cpu(signal, sample_rate)
    raise RuntimeError(f'Unsupported feature backend: {feature_backend}')


def _compute_from_path(file_path: Path, decode_backend: str, feature_backend: str) -> dict[str, Any]:
    signal, sample_rate, decode_used = _decode_audio_file(file_path, decode_backend)
    bundle = _compute_feature_bundle(signal, sample_rate, feature_backend)
    json_ready = feature_bundle_to_json_ready(bundle, source_audio=file_path.name)
    json_ready['decode_backend'] = decode_used
    json_ready['feature_backend_requested'] = feature_backend.lower()
    return json_ready


@app.get('/api/health')
def health() -> dict[str, Any]:
    backend_error = get_cuda_backend_error()
    return {
        'status': 'ok',
        'cuda_backend_available': backend_error == '',
        'cuda_backend_error': backend_error,
        'cuda_backend_path': str(get_default_cuda_library_path()),
        'dali_available': dali_available(),
        'samples': sorted(_sample_lookup().keys()),
    }


@app.get('/api/sample/features')
def sample_features(
    name: str = Query('french_ballet_class.wav'),
    decode_backend: str = Query('auto'),
    feature_backend: str = Query('gpu'),
) -> dict[str, Any]:
    sample_path = _sample_lookup().get(name)
    if sample_path is None:
        raise HTTPException(status_code=404, detail=f'Sample not found: {name}')
    try:
        return _compute_from_path(sample_path, decode_backend, feature_backend)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/api/features')
async def upload_features(
    file: UploadFile = File(...),
    decode_backend: str = Query('auto'),
    feature_backend: str = Query('gpu'),
) -> dict[str, Any]:
    suffix = Path(file.filename or 'upload.wav').suffix or '.wav'
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            payload = await file.read()
            tmp.write(payload)
            temp_path = Path(tmp.name)
        try:
            return _compute_from_path(temp_path, decode_backend, feature_backend)
        finally:
            temp_path.unlink(missing_ok=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get('/')
def root_index() -> FileResponse:
    return FileResponse(APP_DIR / 'index.html')


app.mount('/data', StaticFiles(directory=DATA_DIR), name='data')
app.mount('/', StaticFiles(directory=APP_DIR, html=True), name='frontend')
