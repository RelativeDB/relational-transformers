# Evaluation

Evaluators are callables: construct one with examples, call it with a model, and read
back a metric dictionary. The
[Training Overview](../relational_transformer/training_overview.md#evaluator) shows where
they fit in a training run.

## BinaryClassificationEvaluator

```{eval-rst}
.. autoclass:: relational_transformers.BinaryClassificationEvaluator
   :members:
```

## RegressionEvaluator

```{eval-rst}
.. autoclass:: relational_transformers.RegressionEvaluator
   :members:
```

## SequentialEvaluator

```{eval-rst}
.. autoclass:: relational_transformers.SequentialEvaluator
   :members:
```
