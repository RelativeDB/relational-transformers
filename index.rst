Relational Transformers
=======================

A relational transformer predicts a missing cell from the related data around
it. For a churn prediction, useful evidence might sit in the customer row or
in orders connected by foreign keys. Because the model receives this context
in its original relational shape, each prediction follows the structure
already present in the database.

RT-J is a pretrained relational transformer with 85 million parameters. It
learned from hundreds of databases in The Join, where schemas span fields such
as commerce, sports, finance, and healthcare. Pretraining hides known cells
and asks the model to reconstruct their values from the surrounding context.
At prediction time, the requested value occupies the same masked target
position.

How a cell becomes a token
--------------------------

Your application creates a vector for each cell. With text, the RT-J checkpoint
expects an embedding for the column name beside another embedding for the
value. Scalar channels carry numbers and timestamps, with the semantic type
selecting an input layer before projection into the 512-wide hidden space.

Column names carry meaning across schemas. When two names occupy a compatible
embedding space, a model that learned from ``review_sentiment`` can use that
signal with a new ``customer_mood`` column. Your encoder supplies the space
described in the checkpoint's model card.

How relationships guide attention
---------------------------------

Foreign keys determine which tokens can exchange information. Inside a record,
one attention mask connects its fields; relational masks then route evidence
along references in either direction. Repeated blocks carry information farther
across the database graph while the attention pattern stays sparse.

For each target, the application gathers a bounded context, usually between
256 and 8,192 cells. Starting from the target record, useful joins fill the
available budget. RelativeDB supplies retrieval and tensor construction for
database workloads. Once the vectors and relations are ready, this library
handles prediction, batching, ablation, and training.

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
