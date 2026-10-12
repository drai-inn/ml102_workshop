#!/usr/bin/env bash
# Create a Python venv and Jupyter kernel for the ML102 workshop notebooks on Coder.
#
# Why not requirements.lock.txt: it is a 2022 NeSI snapshot (Python 3.10, TF 2.8,
# CUDA 11 from modules) that does not install on current Python, and its
# git+https://github.com/tensorflow/examples.git line builds a wheel whose version
# embeds the commit hash as a 49-digit integer, which uv refuses to parse.
#
# What this installs instead:
#   - Python 3.11 (the newest TF 2.15 supports)
#   - TensorFlow 2.15.1 with pip-bundled CUDA: the last release with Keras 2,
#     which the notebooks are written against (tf.losses.Reduction, SavedModel
#     load_model, ...)
#   - tensorflow-metadata < 1.18: newer releases import google.protobuf.runtime_version
#     (protobuf 5+), but TF 2.15 requires protobuf 4, so `import tensorflow_datasets`
#     fails (03_transfer_learning.ipynb, segmentation.ipynb)
#   - tensorflow-examples via pip (not uv), pinned to a known commit; only
#     segmentation.ipynb needs it. It is installed last because once it is in the
#     venv, uv refuses to read the venv at all: use `python -m pip`, not
#     `uv pip`, for any later changes.
#   - importlib_resources: tfds.load imports it but does not declare it, so
#     03_transfer_learning.ipynb fails with ModuleNotFoundError without it
#   - a Jupyter kernel "ML102 (TF 2.15)" with:
#       CUDA_CACHE_MAXSIZE=4 GiB: TF 2.15 has no Blackwell (compute capability 12.0)
#         kernels, so the driver compiles them from PTX on first use; the default
#         cache is too small to keep them.
#       TF_FORCE_GPU_ALLOW_GROWTH=true: by default each kernel reserves ~96 of the
#         98 GB of GPU memory, so a second open notebook fails. With growth on,
#         each takes what it uses (1-2 GB for these notebooks).
#
# What this cannot fix (notebook code, not environment):
#   - 03_transfer_learning.ipynb: tfds.load(..., data_dir='/var/lib/tensorflow_datasets')
#     does not exist on Coder and cannot be created without root.
#   - segmentation.ipynb: data_dir is a NeSI path, and "oxford_iiit_pet:3.*.*" no
#     longer exists in tfds 4.9 (only 4.0.0).
#   Point data_dir at a writable folder (or remove it) and use oxford_iiit_pet:4.*.*.
#   First download is slow: oxford_iiit_pet comes from Oxford at ~150 KB/s (~1 h).
#
# Usage:  ./setup_venv.sh [--force]
#   VENV_DIR, KERNEL_NAME, KERNEL_DISPLAY env vars override the defaults.
#   --force deletes an existing VENV_DIR first.
#   The kernel (default: ml102) is registered after the venv is built and is
#   always rewritten to run this venv's python; the script verifies it.

set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-venv3}"
KERNEL_NAME="${KERNEL_NAME:-ml102}"
KERNEL_DISPLAY="${KERNEL_DISPLAY:-ML102 (TF 2.15)}"
TF_EXAMPLES_REF="fa3f48c8b2547ff21eaaa83c1425400f80ecc161"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

command -v uv >/dev/null || { echo "error: uv not found on PATH" >&2; exit 1; }

if [[ -e "$VENV_DIR" ]]; then
  if [[ $FORCE -eq 1 ]]; then
    echo "==> Removing existing $VENV_DIR"
    rm -rf "$VENV_DIR"
  else
    echo "error: $VENV_DIR already exists; rerun with --force to recreate it" >&2
    exit 1
  fi
fi

echo "==> Creating $VENV_DIR (Python 3.11)"
uv venv --python 3.11 "$VENV_DIR"
PY="$(cd "$VENV_DIR" && pwd)/bin/python"

echo "==> Installing TensorFlow 2.15.1 and notebook dependencies"
uv pip install --python "$PY" \
  "tensorflow[and-cuda]==2.15.1" \
  "tensorflow-datasets<4.10" \
  "tensorflow-metadata<1.18" \
  "numpy<2" \
  importlib_resources \
  matplotlib pillow ipython ipywidgets ipykernel pydot pip

echo "==> Installing tensorflow-examples with pip (uv cannot parse its version)"
"$PY" -m pip install --no-deps \
  "git+https://github.com/tensorflow/examples.git@${TF_EXAMPLES_REF}"

# The kernel is registered only now, after the venv is fully installed, and is
# rewritten from scratch every run so it always points at this venv's python.
KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$KERNEL_NAME"
echo "==> Registering Jupyter kernel '$KERNEL_NAME' -> $PY"
rm -rf "$KERNEL_DIR"
"$PY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"
"$PY" - "$KERNEL_DIR/kernel.json" "$PY" <<'EOF'
import json, pathlib, sys
p, py = pathlib.Path(sys.argv[1]), sys.argv[2]
k = json.loads(p.read_text())
k["argv"][0] = py  # pin to the new venv, whatever ipykernel recorded
env = k.setdefault("env", {})
env["CUDA_CACHE_MAXSIZE"] = "4294967296"
env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
p.write_text(json.dumps(k, indent=1))
print(f"    {p}\n    python = {k['argv'][0]}\n    env = {k['env']}")
EOF

# Start the interpreter exactly as Jupyter will, and confirm it is this venv.
KERNEL_PY=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["argv"][0])' "$KERNEL_DIR/kernel.json")
KERNEL_PREFIX=$("$KERNEL_PY" -c 'import sys; print(sys.prefix)')
VENV_ABS="$(cd "$VENV_DIR" && pwd)"
if [[ "$KERNEL_PREFIX" != "$VENV_ABS" ]]; then
  echo "error: kernel '$KERNEL_NAME' runs $KERNEL_PREFIX, expected $VENV_ABS" >&2
  exit 1
fi
echo "    kernel '$KERNEL_NAME' verified: runs $VENV_ABS"

echo "==> Checking the install"
TF_CPP_MIN_LOG_LEVEL=2 "$PY" - <<'EOF'
import tensorflow as tf, keras, tensorflow_datasets as tfds, importlib_resources
from tensorflow_examples.models.pix2pix import pix2pix
print(f"    tensorflow {tf.__version__}, keras {keras.__version__}, tfds {tfds.__version__}")
print(f"    GPUs: {tf.config.list_physical_devices('GPU') or 'none (will run on CPU)'}")
EOF

cat <<EOF

Done. In JupyterLab, open a notebook and pick the "$KERNEL_DISPLAY" kernel
(the notebooks default to "Python 3 (ipykernel)", which has no TensorFlow).

Notes:
  - The first training epoch of each new model is slow while the GPU code is
    compiled and cached; later runs reuse the cache.
  - 03_transfer_learning.ipynb and segmentation.ipynb load datasets from
    paths that do not exist on Coder (/var/lib/tensorflow_datasets, /nesi/...).
    Edit data_dir in those cells, and use "oxford_iiit_pet:4.*.*" in
    segmentation.ipynb. The first oxford_iiit_pet download takes about an hour.
  - To add packages later use "$VENV_DIR/bin/python -m pip install ...":
    uv cannot read this venv once tensorflow-examples is installed.
EOF
