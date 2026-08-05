# Losses

Which loss fits which problem type, along with output and label shapes, is covered in the
[Loss Overview](../relational_transformer/loss_overview.md). Every module here takes raw
logits and works in ordinary PyTorch loops.

## BinaryClassificationLoss

```{eval-rst}
.. autoclass:: relational_transformers.BinaryClassificationLoss
   :members:
```

## MulticlassClassificationLoss

```{eval-rst}
.. autoclass:: relational_transformers.MulticlassClassificationLoss
   :members:
```

## MultilabelClassificationLoss

```{eval-rst}
.. autoclass:: relational_transformers.MultilabelClassificationLoss
   :members:
```

## ListwiseRankingLoss

```{eval-rst}
.. autoclass:: relational_transformers.ListwiseRankingLoss
   :members:
```

## RegressionLoss

```{eval-rst}
.. autoclass:: relational_transformers.RegressionLoss
   :members:
```

## loss_for

```{eval-rst}
.. autofunction:: relational_transformers.losses.loss_for
```
