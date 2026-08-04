# Installation

Relational Transformers requires Python 3.10 or newer.

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

For an editable source checkout:

```bash
git clone https://github.com/RelativeDB/relational-transformers
cd relational-transformers
python -m pip install -e '.[dev]'
pytest
```

Model weights download from Hugging Face on first use and remain in its normal
local cache. A local directory with `config.json` and `model.safetensors` uses
the same constructor and never accesses the network.
