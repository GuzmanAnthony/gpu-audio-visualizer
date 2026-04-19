#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing virtual environment at $ROOT_DIR/.venv"
  echo "Create it first, then run: source grace_env.sh"
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

PYREAL="$(readlink -f "$VENV_PY")"
PYBASE="$(dirname "$(dirname "$PYREAL")")"
LIB_PATHS=()

if [[ -d "$PYBASE/lib" ]]; then
  LIB_PATHS+=("$PYBASE/lib")
fi

if command -v nvcc >/dev/null 2>&1; then
  CUDA_BIN="$(dirname "$(readlink -f "$(command -v nvcc)")")"
  CUDA_ROOT="$(dirname "$CUDA_BIN")"
  if [[ -d "$CUDA_ROOT/lib64" ]]; then
    LIB_PATHS+=("$CUDA_ROOT/lib64")
  fi
fi

LIBFFI_DIR="$(find /sw/eb/sw -maxdepth 4 -type f -name 'libffi.so*' 2>/dev/null | head -n 1 | xargs -r dirname || true)"
if [[ -n "$LIBFFI_DIR" && -d "$LIBFFI_DIR" ]]; then
  LIB_PATHS+=("$LIBFFI_DIR")
fi

for candidate in "${LIB_PATHS[@]}"; do
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$candidate:"*) ;;
    *) export LD_LIBRARY_PATH="$candidate${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
done

export PYTHONNOUSERSITE=1

echo "Environment ready."
echo "Python: $(python --version 2>&1)"
echo "nvcc: $(nvcc --version | tail -n 1 2>/dev/null || echo 'not found')"
