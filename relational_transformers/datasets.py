"""Small dataset containers for relational examples."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from torch.utils.data import Dataset

from .training import RelationalExample


class RelationalDataset(Dataset):
    """A validated, indexable collection of :class:`RelationalExample` objects."""

    def __init__(self, examples: Iterable[RelationalExample]) -> None:
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("RelationalDataset requires at least one example")
        if not all(isinstance(example, RelationalExample) for example in self.examples):
            raise TypeError("every dataset item must be a RelationalExample")

    @classmethod
    def from_inputs(cls, inputs: Sequence, labels: Sequence) -> RelationalDataset:
        """Create a dataset from aligned model inputs and labels."""

        if len(inputs) != len(labels):
            raise ValueError("inputs and labels must have the same length")
        return cls(
            RelationalExample(value, label) for value, label in zip(inputs, labels, strict=True)
        )

    def __getitem__(self, index: int) -> RelationalExample:
        return self.examples[index]

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[RelationalExample]:
        return iter(self.examples)
