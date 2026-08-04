# Custom models

## Checkpoint Resolution

The constructor's first argument resolves in three ways:

1. **A file path** loads those weights directly, reading `config.json` from the same
   directory when present.
2. **A directory** loads `<directory>/config.json` and the weights file it names through
   `checkpoint_file`, which defaults to `model.safetensors`. When the directory contains
   `classification/` or `regression/` subfolders, `task=` picks the subfolder.
3. **Anything else** is treated as a Hugging Face repository ID and downloaded through
   `huggingface_hub`, with the same `task=` subfolder rule and an optional `revision=`.

The configuration declares dimensions either as top-level keys or nested under a
`"model"` key:

```json
{
  "task_type": "clf",
  "model": {"num_blocks": 12, "d_model": 512, "d_text": 384, "num_heads": 8, "d_ff": 2048},
  "checkpoint_file": "model.safetensors"
}
```

Missing dimension keys raise `ValueError` at load time. `.pt` checkpoints load through
`torch.load` as a fallback; safetensors is the published format.

## Saving Checkpoints

`save_pretrained` writes `model.safetensors` and `config.json` in the layout described
above, so a fine-tuned model reloads with the same constructor:

```python
model.save_pretrained("models/churn-v2")
reloaded = RelationalTransformer("models/churn-v2")
```

## Working with RTJModel Directly

The public `RTJModel` is an ordinary `torch.nn.Module` whose state-dict names match
published RelativeDB RT-J checkpoints.
`model.model` exposes the loaded instance on the torch backend, and the class constructs
standalone for training from scratch:

```python
from relational_transformers import RTJModel

small = RTJModel(num_blocks=2, d_model=64, d_text=16, num_heads=4, d_ff=128)
output = small(batch, output="target_scores")
```

Advanced users can replace decoder heads, freeze blocks, register hooks, or run their own
optimization and mixed-precision setup over it. Use the meta backend when inspecting or
transforming a large architecture without allocating weights.

```{eval-rst}
.. collapse:: Module layout

   - ``enc_dict``: value encoders for ``number``, ``text``, ``datetime``, ``boolean``, and ``col_name``
   - ``dec_dict``: decoders back to each semantic channel; ``number`` produces target scores
   - ``norm_dict`` and ``norm_out``: RMS normalization around encoders and the final state
   - ``mask_embs``: one learned target-mask embedding per semantic type
   - ``blocks``: the stack of relational blocks, each with ``col``, ``feat``, and ``nbr``
     attention plus a feed-forward layer
```

## Changing the Embedding Space

Custom input encoders must preserve the checkpoint contract or be trained jointly with
the backbone. A checkpoint trained against `all-MiniLM-L12-v2` reads any other 384-wide
embedding space as noise, because two encoders of the same width still place meanings at
unrelated coordinates. Options
that work: fine-tune the full model on data encoded your way, or train an adapter that
maps your encoder's space into the checkpoint's.
