#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

export GPU_AUDIO_CUDA_LIB="${GPU_AUDIO_CUDA_LIB:-$ROOT_DIR/cuda_backend/build/libgpuaudio_features.so}"

if command -v nvcc >/dev/null 2>&1; then
  CUDA_BIN_DIR="$(dirname "$(command -v nvcc)")"
  CUDA_ROOT="$(cd "$CUDA_BIN_DIR/.." && pwd)"

  for CUDA_LIB_DIR in \
    "$CUDA_ROOT/lib64" \
    "$CUDA_ROOT/targets/x86_64-linux/lib" \
    "$CUDA_ROOT/lib"
  do
    if [ -d "$CUDA_LIB_DIR" ]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":$CUDA_LIB_DIR:"*) ;;
        *) export LD_LIBRARY_PATH="$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}" ;;
      esac
    fi
  done
fi

echo "Using CUDA feature library: $GPU_AUDIO_CUDA_LIB"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

uvicorn server:app --host "$HOST" --port "$PORT"