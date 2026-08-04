# Installation

Relational Transformers requires Python 3.10 or newer.

## Install with uv

```bash
uv add relational-transformers
```

Add deployment extras with `uv add 'relational-transformers[onnx]'` or
`uv add 'relational-transformers[triton]'`.

## Install with pip

```bash
pip install -U relational-transformers
```

PyTorch is the default backend and supports CPU, Apple MPS, and CUDA. Install
deployment extras only where they are used:

```bash
pip install -U 'relational-transformers[triton]'
pip install -U 'relational-transformers[onnx]'
pip install -U 'relational-transformers[dev]'
```

Cell encoders are intentionally application-owned. The quickstart uses
Sentence Transformers to reproduce the released RT-J embedding space, so its
example environment also installs it explicitly:

```bash
pip install -U sentence-transformers
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

For development, install the checkout with test and documentation dependencies:

```bash
git clone https://github.com/RelativeDB/relational-transformers
cd relational-transformers
python -m pip install -e '.[dev]'
pytest
```

## Install PyTorch with CUDA support

Install the PyTorch build matching the CUDA runtime on the deployment host,
then install `relational-transformers[triton]`. Follow the current command from
[PyTorch's installation selector](https://pytorch.org/get-started/locally/)
rather than pinning a CUDA wheel URL in application code.

Model weights download from Hugging Face on first use and remain in its normal
local cache. A local directory with `config.json` and `model.safetensors` uses
the same constructor and never accesses the network.
