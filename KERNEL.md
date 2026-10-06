# The workshop's Jupyter kernel

The notebooks run in a Jupyter kernel backed by a Python virtual environment
inside the repository. [`setup_venv.sh`](setup_venv.sh) builds both. This page
is for maintainers and anyone who needs to change the environment. To just run
the workshop, follow the [Setup](README.md#setup) section of the README.

## Requirements

- A Coder workspace with JupyterLab.
- [`uv`](https://docs.astral.sh/uv/) on `PATH`. It is preinstalled in the
  workspace image. The script exits with an error if it is missing.
- Internet access, to download Python 3.11 and about 4.7 GB of packages.

A GPU is optional. Without one, the notebooks run on the CPU, which is much
slower.

## Usage

```
./setup_venv.sh [--force]
```

| Option | Default | Meaning |
|---|---|---|
| `--force` | off | Delete an existing virtual environment first. Without it, the script refuses to overwrite one. |
| `VENV_DIR` | `venv3` | Where to create the virtual environment, relative to the repository. |
| `KERNEL_NAME` | `ml102` | Jupyter's internal name for the kernel. The notebooks are saved with this name, so they select the kernel automatically. If you change it, you will have to pick the kernel by hand in each notebook. |
| `KERNEL_DISPLAY` | `ML102 (TF 2.15)` | The name shown in JupyterLab. |

Environment variables go before the command, for example
`VENV_DIR=venv-test KERNEL_NAME=ml102-test ./setup_venv.sh`.

The script always works relative to its own folder, so you can run it from
anywhere.

## What it installs

| Package | Version | Why |
|---|---|---|
| Python | 3.11 | The newest Python that TensorFlow 2.15 supports. |
| `tensorflow[and-cuda]` | 2.15.1 | The last TensorFlow release that uses Keras 2, which the notebooks are written for (for example `tf.losses.Reduction` and loading SavedModels with `load_model`). `[and-cuda]` installs CUDA and cuDNN from pip, so no system CUDA is needed. |
| `tensorflow-datasets` | < 4.10 | Compatible with TensorFlow 2.15. |
| `tensorflow-metadata` | < 1.18 | Newer releases need protobuf 5, but TensorFlow 2.15 needs protobuf 4. With a newer version, `import tensorflow_datasets` fails. |
| `numpy` | < 2 | TensorFlow 2.15 is built against NumPy 1. |
| `importlib_resources` | latest | `tfds.load` imports it without declaring it as a dependency, so `03_transfer_learning.ipynb` fails without it. |
| `matplotlib`, `pillow`, `ipython`, `ipywidgets`, `ipykernel`, `pydot`, `pip` | latest | Plotting, images, notebook widgets, the kernel itself, and `plot_model` in `segmentation.ipynb`. |
| `tensorflow-examples` | commit `fa3f48c` | Provides `pix2pix`, used only by `segmentation.ipynb`. Installed from GitHub with pip, not uv, because its version string embeds the commit hash as a 49-digit number that uv cannot parse. |

It then registers the kernel in `~/.local/share/jupyter/kernels/<KERNEL_NAME>/`
with two environment variables set:

- `CUDA_CACHE_MAXSIZE=4294967296` (4 GiB). TensorFlow 2.15 has no compiled
  code for the Blackwell GPUs (compute capability 12.0) in these workspaces,
  so the driver compiles it on first use and caches the result. The default
  cache is too small to hold it, which would make every run as slow as the
  first.
- `TF_FORCE_GPU_ALLOW_GROWTH=true`. By default, TensorFlow reserves almost all
  GPU memory as soon as a notebook touches the GPU, so a second notebook
  fails. With this set, each notebook takes only what it uses (1-2 GB here).

The kernel definition is deleted and rewritten on every run, so it always
points at the virtual environment the script just built.

## Checks

Before it finishes, the script:

1. starts the kernel's Python exactly as Jupyter will and confirms it runs
   from the new virtual environment, and
2. imports `tensorflow`, `keras`, `tensorflow_datasets`, `importlib_resources`
   and `pix2pix`, then prints the versions and the GPUs TensorFlow can see.

If either check fails, the script exits with an error. A line reading
`GPUs: none (will run on CPU)` is not an error, but it means the notebooks
will be slow.

## Adding packages

Once `tensorflow-examples` is installed, uv refuses to read the virtual
environment at all (the same version-string problem as above). Use pip
instead:

```
venv3/bin/python -m pip install <package>
```

If the package is needed by the notebooks, add it to the `uv pip install`
line in `setup_venv.sh` too, so the next rebuild includes it.

## Rebuilding and removing

To rebuild from scratch, for example after editing the script:

```
./setup_venv.sh --force
```

To remove the environment and the kernel:

```
rm -rf ~/.local/share/jupyter/kernels/ml102
rm -rf venv3
```

## Troubleshooting

- **The notebook says `No module named 'tensorflow'`.** It is using the
  default **Python 3 (ipykernel)** kernel. Switch to **ML102 (TF 2.15)**.
- **The kernel is missing from JupyterLab.** Reload the browser page. If it
  is still missing, check that
  `~/.local/share/jupyter/kernels/ml102/kernel.json` exists, and rerun the
  script with `--force`.
- **The kernel dies as soon as it starts, or never connects.** The kernel
  definition records the full path to the virtual environment's Python. If
  the repository folder has been moved or renamed since the script ran, that
  path no longer exists. Check the first line of `argv` in
  `~/.local/share/jupyter/kernels/ml102/kernel.json`, and rerun the script
  with `--force` from the repository's current location.
- **`CUDA_ERROR_OUT_OF_MEMORY` or failure to create a GPU context.** Another
  kernel may be holding the GPU. Shut down kernels you are not using, or check
  with `nvidia-smi`.
- **Every training run is as slow as the first.** The compiled GPU code is
  not being cached. Check that `CUDA_CACHE_MAXSIZE` is in the kernel's
  `kernel.json`.
