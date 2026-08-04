# Custom models

The public `RTJModel` is an ordinary `torch.nn.Module` whose state-dict names
match published RT-J checkpoints. Advanced users can replace decoder heads,
freeze blocks, register hooks, compile the model, or train from a new
initialization.

Use the meta backend when inspecting or transforming a large architecture
without allocating weights. Custom input encoders must preserve the checkpoint
contract or be trained jointly with the backbone.
