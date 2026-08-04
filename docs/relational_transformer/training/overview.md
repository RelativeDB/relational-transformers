# Training overview

There are two adaptation paths:

- **Head tuning** encodes every target once, freezes RT-J, and trains a small
  classifier or regressor over the 512-wide target feature.
- **Full fine-tuning** differentiates through value encoders, relational
  attention blocks, normalization, and decoder heads.

Both paths consume `RelationalExample(input=batch, label=...)` and use the same
inference tensor contract. Split data by time before constructing examples to
avoid future context leaking into training features or normalization.
