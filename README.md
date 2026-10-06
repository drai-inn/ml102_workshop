# ML102 - Image analysis

A hands-on introduction to image analysis with neural networks, using
TensorFlow and Keras in JupyterLab on a Coder workspace with a GPU.

This is an adaptation of the NeSI/REANNZ
[ML102 workshop](https://github.com/nesi/ml102_workshop), which ran on the
REANNZ Open OnDemand platform. The notebooks have been changed to run in a
Coder workspace, and a setup script now builds the Python environment for you.

## Setup

You only need to do this once per workspace.

1. Start your Coder workspace and open a terminal (in JupyterLab:
   **File > New > Terminal**).

2. Get the workshop files, if they are not already in your home folder:

   ```
   cd ~
   git clone https://github.com/drai-inn/ml102_workshop.git
   cd ml102_workshop
   ```

3. Run the setup script:

   ```
   ./setup_venv.sh
   ```

   It creates a Python environment in `ml102_workshop/venv3`, installs
   TensorFlow 2.15 and the other packages the notebooks need, and adds a
   Jupyter kernel called **ML102 (TF 2.15)**. It downloads about 4.7 GB, so
   it takes a few minutes. At the end it prints the TensorFlow version and the
   GPUs it can see.

   If `venv3` already exists, the script stops rather than overwrite it. Run
   `./setup_venv.sh --force` to delete it and start again.

4. Reload JupyterLab in your browser so it picks up the new kernel.

5. Open a notebook from the `notebooks/` folder. It should start with the
   **ML102 (TF 2.15)** kernel, shown at the top right of the notebook. If it
   says **Python 3 (ipykernel)** instead, click the kernel name and choose
   **ML102 (TF 2.15)**. The default Python kernel has no TensorFlow.

More detail on what the script installs, and how to change it, is in
[KERNEL.md](KERNEL.md).

## Workshop

The workshop is a series of notebooks, adapted from the
[TensorFlow tutorials](https://www.tensorflow.org/tutorials):

1. [Introduction](notebooks/01_introduction.ipynb)
1. [Image classification](notebooks/02_classification.ipynb) ([source](https://www.tensorflow.org/tutorials/images/classification))
1. [Transfer learning and fine-tuning](notebooks/03_transfer_learning.ipynb) ([source](https://www.tensorflow.org/tutorials/images/transfer_learning))

## Supplemental material

These notebooks are not part of the main workshop:

- [Convolutional Neural Network (CNN)](notebooks/cnn.ipynb) example ([source](https://www.tensorflow.org/tutorials/images/cnn))
- [Image Segmentation](notebooks/segmentation.ipynb) ([source](https://www.tensorflow.org/tutorials/images/segmentation))

## Things to know

- **Datasets are downloaded the first time you run a notebook.** Later runs
  reuse the copy on disk.

  | Notebook | Dataset | Saved to |
  |---|---|---|
  | `02_classification` | flower photos | `~/.keras/datasets/` |
  | `03_transfer_learning` | cats_vs_dogs | `~/ml102_workshop/tensorflow_datasets/` |
  | `cnn` | CIFAR-10 | `~/.keras/datasets/` |
  | `segmentation` | oxford_iiit_pet | `~/ml102_workshop/tensorflow_datasets/` |

  The oxford_iiit_pet download for `segmentation.ipynb` is slow (about an
  hour), so start it well before you need it.

- **The first training run is slow.** TensorFlow 2.15 predates the GPU in
  these workspaces, so the GPU driver compiles TensorFlow's code on first use
  and caches it. You will see warnings like `TensorFlow was not built with
  CUDA kernel binaries compatible with compute capability 12.0`. They are
  expected and harmless. Later runs reuse the cache and are much faster.

- **Several notebooks can share the GPU.** The kernel lets each notebook take
  only the GPU memory it needs, so you can keep more than one open. To free
  the memory, shut down the kernels you are not using
  (**Kernel > Shut Down Kernel**).

- **Installing more packages:** use
  `~/ml102_workshop/venv3/bin/python -m pip install <package>`, not
  `uv pip install`. See [KERNEL.md](KERNEL.md#adding-packages) for why.
