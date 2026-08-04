# Installation

Relational Transformers requires Python 3.10 or newer and PyTorch 2.2 or newer. The base
package installs the portable PyTorch runtime, which serves inference and training on CPU,
Apple MPS, and CUDA devices. Everything else ships as an extra:

- **`relational-transformers`**: the default. PyTorch inference, training, evaluation, and checkpoint tools.
- **`relational-transformers[onnx]`**: adds ONNX export and ONNX Runtime inference.
- **`relational-transformers[triton]`**: adds the optimized Triton CUDA serving backend.
- **`relational-transformers[dev]`**: adds pytest, coverage, ruff, and the ONNX toolchain for development.
- **`relational-transformers[docs]`**: adds Sphinx and the theme used to build this documentation.

## Install with uv

```{eval-rst}
.. tab:: Default

   ::

      uv add relational-transformers

.. tab:: ONNX

   ::

      uv add 'relational-transformers[onnx]'

.. tab:: Triton

   ::

      uv add 'relational-transformers[triton]'

.. tab:: Development

   ::

      uv add 'relational-transformers[dev]'
```

## Install with pip

```{eval-rst}
.. tab:: Default

   ::

      pip install -U relational-transformers

.. tab:: ONNX

   ::

      pip install -U 'relational-transformers[onnx]'

.. tab:: Triton

   ::

      pip install -U 'relational-transformers[triton]'

.. tab:: Development

   ::

      pip install -U 'relational-transformers[dev]'
```

Install deployment extras only on the hosts that use them. A CPU inference host has no use
for the Triton kernels, and an export pipeline needs `[onnx]` while the serving host that
loads the exported file needs only `onnxruntime`.

```{eval-rst}
.. tip::

   Cell encoders are application-owned, so no encoder is installed automatically. The
   quickstart reproduces the released RT-J embedding space with Sentence Transformers,
   which its example environment installs explicitly::

      pip install -U relational-transformers sentence-transformers
```

## Install with Conda

Create an isolated environment with Conda, then install the package from PyPI:

```bash
conda create -n relational-transformers python=3.12
conda activate relational-transformers
python -m pip install -U relational-transformers
```

## Install from Source

```bash
git clone https://github.com/RelativeDB/relational-transformers
cd relational-transformers
python -m pip install .
```

## Editable Install

For development, install the checkout in editable mode with the test dependencies, then run
the suite to confirm the environment works:

```bash
git clone https://github.com/RelativeDB/relational-transformers
cd relational-transformers
python -m pip install -e '.[dev]'
pytest
```

The default test suite is deterministic and runs offline on CPU. The [Testing](testing.md)
guide describes the opt-in checkpoint and CUDA suites.

## Install PyTorch with CUDA support

Install the PyTorch build matching the CUDA runtime on the deployment host, then install
`relational-transformers[triton]`. Follow the current command from
[PyTorch's installation selector](https://pytorch.org/get-started/locally/); a CUDA wheel
URL pinned in application code goes stale with the next driver rollout.

## Model Files and Caching

`RelationalTransformer()` downloads the default `RelativeDB/rt-j-fp16` configuration
and its declared weights file from the Hugging Face Hub on first use, then keeps them in
the normal `huggingface_hub` cache. Later constructions read from the cache without
network access.

A local checkpoint works through the same constructor and never touches the network:

```python
from relational_transformers import RelationalTransformer

model = RelationalTransformer("/models/rt-j-fp16")
```

The directory needs `config.json` plus the weights file it names, which defaults to
`model.safetensors`. Published repositories keep `classification/` and `regression/`
subfolders; a local directory may use the same layout or hold a single checkpoint at its
root. See [Custom Models](relational_transformer/usage/custom_models.md) for the full
resolution rules.
