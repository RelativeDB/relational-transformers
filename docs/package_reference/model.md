# Model API

`RelationalTransformer` is the high-level entry point for loading checkpoints and
computing predictions; the [Quickstart](../quickstart.md) shows it in context. `RTJModel`
is the underlying `torch.nn.Module` for custom training loops and architecture work, and
`ModelOutput` carries every result field.

## RelationalTransformer

```{eval-rst}
.. autoclass:: relational_transformers.RelationalTransformer
   :members:
```

## RTJModel

```{eval-rst}
.. autoclass:: relational_transformers.RTJModel
   :members:
```

## ModelOutput

```{eval-rst}
.. autoclass:: relational_transformers.ModelOutput
   :members:
```
