#!/usr/bin/env bash
# One-shot setup on a fresh Radeon Cloud instance.
#
# Assumes the AMD ROCm PyTorch template (torch is already installed and sees the GPU).
# Installs everything else this project needs and verifies the machine can run the whole
# pipeline — simulation included — before you spend GPU hours on it.
#
#   bash setup.sh
#
set -euo pipefail

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

cd "$(dirname "$0")"

log "1/5  System libraries (Genesis renders offscreen; ffmpeg decodes dataset video)"
if command -v apt-get >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 libosmesa6 libegl1 git
fi

log "2/5  Confirming the template's torch sees the Radeon GPU"
python - <<'PY'
import sys, torch
print("torch:", torch.__version__)
ok = torch.cuda.is_available()
print("GPU available:", ok, "|", torch.cuda.get_device_name(0) if ok else "no device")
if not ok:
    sys.exit("ABORT: torch cannot see the GPU. Use the ROCm template, and do NOT pip install torch "
             "(the PyPI wheel is the CUDA build).")
PY

log "3/5  Python dependencies"
# torch is deliberately absent from requirements.txt — see the file's comments.
# torchcodec is never installed: its compiled extension is ABI-incompatible with AMD's
# custom torch build, so every command here passes --dataset.video_backend=pyav.
pip install --quiet --no-cache-dir -r requirements.txt
pip uninstall -y -q torchcodec 2>/dev/null || true

log "4/5  Environment defaults"
# Genesis rasterises the two cameras through pyrender, whose default pyglet backend needs a
# display and dies headless with "IndexError: list index out of range". EGL renders without
# one. Persist it so later shells inherit it.
ENVLINE='export PYOPENGL_PLATFORM=egl'
grep -qxF "$ENVLINE" ~/.bashrc 2>/dev/null || echo "$ENVLINE" >> ~/.bashrc
export PYOPENGL_PLATFORM=egl
echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM (also appended to ~/.bashrc)"

log "5/5  Smoke test — torch+ROCm, EGL, scene build, both cameras rendering"
python src/smoke_test_rocm.py

log "Setup complete. Next: bash run_pipeline.sh --quick"
