Relational Transformers
=======================

A relational transformer predicts a missing cell from the related data around
it. For a churn prediction, useful evidence might sit in the customer row or
in orders connected by foreign keys. The model receives this context in its
original relational shape, so each prediction follows the structure already
present in the database.

RT-J is a pretrained relational transformer with 85 million parameters. It
learned from hundreds of databases in The Join, where schemas span commerce,
sports, finance, and healthcare. Pretraining hides known cells and asks the
model to reconstruct their values from the surrounding context; at prediction
time, the requested value occupies the same masked target position.

Your application owns retrieval and encoding: it gathers a bounded context of
related cells, embeds text and column names with its own encoder, and
normalizes scalars. This library owns everything after that point:
prediction, batching, ablation, and training over the resulting tensors, with
one input contract across the PyTorch, Triton, and ONNX backends. RelativeDB
is the reference integration for database workloads, and
`relational-transformers-utils <https://utils.relationaltransformers.com/>`_
carries the context-construction and measurement tooling.

Using the library
-----------------

The default constructor downloads the published RT-J classification
checkpoint. A model-ready array places the masked target at row zero and keeps
the related cell vectors after it.

.. code-block:: python

   from relational_transformers import RelationalTransformer

   model = RelationalTransformer()
   probability = model.predict(cell_vectors, target=0)

A batch can hold contexts of different lengths. Training and deployment keep
the same input contract across the available backends. In the
:doc:`quickstart <docs/quickstart>`, you build the vectors for a complete
prediction before moving into typed relational batches.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   docs/installation
   docs/testing
   docs/quickstart
   examples/README
   docs/relational_transformer/usage/usage
   docs/relational_transformer/usage/prediction
   docs/relational_transformer/usage/batches
   docs/relational_transformer/usage/backends
   docs/relational_transformer/usage/efficiency
   docs/relational_transformer/usage/ablation
   docs/relational_transformer/usage/custom_models
   docs/relational_transformer/pretrained_models
   docs/relational_transformer/dataset_overview
   docs/relational_transformer/loss_overview
   docs/relational_transformer/training_overview
   docs/relational_transformer/training/overview
   docs/relational_transformer/training/head_tuning
   docs/relational_transformer/training/full_finetuning
   docs/relational_transformer/training/examples
   docs/package_reference/model
   docs/package_reference/batch
   docs/package_reference/datasets
   docs/package_reference/evaluation
   docs/package_reference/losses
   docs/package_reference/training
   docs/package_reference/onnx
