"""S15 native data pipeline and tokenizer references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from . import rng as tessera_rng


class Dataset:
    """Small eager dataset with deterministic transform semantics."""

    def __init__(self, items: Iterable[Any], *, state: Mapping[str, Any] | None = None):
        self._items = list(items)
        self._cursor = int((state or {}).get("cursor", 0))

    @classmethod
    def from_tensor_slices(cls, tensors: Any) -> "Dataset":
        if isinstance(tensors, Mapping):
            n = len(next(iter(tensors.values()))) if tensors else 0
            return cls({k: np.asarray(v)[i] for k, v in tensors.items()} for i in range(n))
        arr = np.asarray(tensors)
        return cls(arr[i] for i in range(len(arr)))

    @classmethod
    def synthetic(cls, count: int, fn: Callable[[int], Any] | None = None) -> "Dataset":
        return cls((fn or (lambda i: i))(i) for i in range(int(count)))

    @classmethod
    def sharded_file_source(cls, path: str | Path) -> "Dataset":
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            keys = sorted(arr.files)
            n = len(arr[keys[0]]) if keys else 0
            return cls({k: arr[k][i] for k in keys} for i in range(n))
        return cls(arr[i] for i in range(len(arr)))

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items[self._cursor:])

    def __len__(self) -> int:
        return max(0, len(self._items) - self._cursor)

    def to_list(self) -> list[Any]:
        return list(iter(self))

    def map(self, fn: Callable[[Any], Any]) -> "Dataset":
        return Dataset(fn(x) for x in self)

    def filter(self, pred: Callable[[Any], bool]) -> "Dataset":
        return Dataset(x for x in self if bool(pred(x)))

    def batch(self, size: int, *, drop_remainder: bool = False) -> "Dataset":
        size = int(size)
        if size <= 0:
            raise ValueError("batch size must be positive")
        batches = []
        cur = []
        for item in self:
            cur.append(item)
            if len(cur) == size:
                batches.append(_stack_batch(cur))
                cur = []
        if cur and not drop_remainder:
            batches.append(_stack_batch(cur))
        return Dataset(batches)

    def prefetch(self, buffer_size: int = 1) -> "Dataset":
        if buffer_size <= 0:
            raise ValueError("prefetch buffer_size must be positive")
        return Dataset(self)

    def shuffle(self, key: tessera_rng.RNGKey, *, buffer_size: int | None = None) -> "Dataset":
        items = self.to_list()
        if buffer_size is not None and buffer_size <= 0:
            raise ValueError("shuffle buffer_size must be positive")
        order = tessera_rng.permutation(key, len(items))
        return Dataset(items[int(i)] for i in order)

    def interleave(self, fn: Callable[[Any], "Dataset"], *, cycle_length: int = 1) -> "Dataset":
        if cycle_length <= 0:
            raise ValueError("cycle_length must be positive")
        streams = [iter(fn(x)) for x in self]
        out = []
        active = list(streams)
        while active:
            nxt = []
            for it in active:
                try:
                    out.append(next(it))
                    nxt.append(it)
                except StopIteration:
                    pass
            active = nxt
        return Dataset(out)

    def take(self, n: int) -> "Dataset":
        return Dataset(self.to_list()[: int(n)])

    def skip(self, n: int) -> "Dataset":
        return Dataset(self.to_list()[int(n):])

    def repeat(self, count: int | None = None) -> "Dataset":
        items = self.to_list()
        if count is None:
            raise ValueError("infinite repeat is not materialized in the reference Dataset")
        return Dataset(items * int(count))

    def concatenate(self, other: "Dataset") -> "Dataset":
        return Dataset([*self.to_list(), *other.to_list()])

    def zip(self, other: "Dataset") -> "Dataset":
        return Dataset(zip(self, other))

    def unbatch(self) -> "Dataset":
        out = []
        for batch in self:
            if isinstance(batch, Mapping):
                n = len(next(iter(batch.values()))) if batch else 0
                out.extend({k: np.asarray(v)[i] for k, v in batch.items()} for i in range(n))
            else:
                out.extend(list(batch))
        return Dataset(out)

    def shard(self, mesh: Any, axis_name: str, index: int = 0) -> "ShardedDataset":
        return ShardedDataset(self, mesh=mesh, axis_name=axis_name, index=index)

    def checkpoint(self) -> dict[str, Any]:
        return {"cursor": int(self._cursor), "length": len(self._items)}

    def restore(self, state: Mapping[str, Any]) -> "Dataset":
        return Dataset(self._items, state=state)

    def replay_from(self, epoch: int, step: int) -> "Dataset":
        return Dataset(self._items, state={"cursor": int(step)})


class IterableDataset:
    """Streaming dataset wrapper with checkpointable cursor."""

    def __init__(self, source: Callable[[], Iterable[Any]], *, cursor: int = 0, prefetch_buffer: int = 1):
        self.source = source
        self.cursor = int(cursor)
        self.prefetch_buffer = int(prefetch_buffer)

    def __iter__(self) -> Iterator[Any]:
        for i, item in enumerate(self.source()):
            if i >= self.cursor:
                yield item

    def map(self, fn: Callable[[Any], Any]) -> "IterableDataset":
        return IterableDataset(lambda: (fn(x) for x in self), prefetch_buffer=self.prefetch_buffer)

    def take(self, n: int) -> Dataset:
        out = []
        for i, item in enumerate(self):
            if i >= int(n):
                break
            out.append(item)
        return Dataset(out)

    def checkpoint(self) -> dict[str, Any]:
        return {"cursor": self.cursor, "prefetch_buffer": self.prefetch_buffer}

    def restore(self, state: Mapping[str, Any]) -> "IterableDataset":
        return IterableDataset(self.source, cursor=int(state.get("cursor", 0)), prefetch_buffer=int(state.get("prefetch_buffer", 1)))


class ShardedDataset(Dataset):
    def __init__(self, dataset: Dataset, *, mesh: Any, axis_name: str, index: int = 0):
        self.mesh = mesh
        self.axis_name = axis_name
        self.index = int(index)
        size = _axis_size(mesh, axis_name)
        super().__init__(item for i, item in enumerate(dataset) if i % size == self.index)


def dataset_map(dataset: Dataset, fn: Callable[[Any], Any]) -> Dataset:
    return dataset.map(fn)


def dataset_filter(dataset: Dataset, pred: Callable[[Any], bool]) -> Dataset:
    return dataset.filter(pred)


def dataset_batch(dataset: Dataset, size: int, *, drop_remainder: bool = False) -> Dataset:
    return dataset.batch(size, drop_remainder=drop_remainder)


def dataset_prefetch(dataset: Dataset, buffer_size: int = 1) -> Dataset:
    return dataset.prefetch(buffer_size)


def dataset_shuffle(dataset: Dataset, key: tessera_rng.RNGKey, *, buffer_size: int | None = None) -> Dataset:
    return dataset.shuffle(key, buffer_size=buffer_size)


def dataset_interleave(dataset: Dataset, fn: Callable[[Any], Dataset], *, cycle_length: int = 1) -> Dataset:
    return dataset.interleave(fn, cycle_length=cycle_length)


def dataset_repeat(dataset: Dataset, count: int) -> Dataset:
    return dataset.repeat(count)


def dataset_zip(a: Dataset, b: Dataset) -> Dataset:
    return a.zip(b)


def sharded_dataset(dataset: Dataset, mesh: Any, axis_name: str, index: int = 0) -> ShardedDataset:
    return dataset.shard(mesh, axis_name, index=index)


def iterable_dataset(source: Callable[[], Iterable[Any]], *, prefetch_buffer: int = 1) -> IterableDataset:
    return IterableDataset(source, prefetch_buffer=prefetch_buffer)


def dataset_checkpoint(dataset: Dataset | IterableDataset) -> dict[str, Any]:
    return dataset.checkpoint()


class Tokenizer:
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def special_tokens(self) -> dict[str, int]:
        return {}


@dataclass
class ByteTokenizer(Tokenizer):
    specials: Mapping[str, int] | None = None

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(int(i) for i in ids).decode("utf-8")

    @property
    def vocab_size(self) -> int:
        return 256

    @property
    def special_tokens(self) -> dict[str, int]:
        return dict(self.specials or {})


class VocabTokenizer(Tokenizer):
    def __init__(self, vocab: Mapping[str, int], *, unk_token: str = "<unk>", specials: Mapping[str, int] | None = None):
        self.vocab = dict(vocab)
        self.unk_token = unk_token
        self._specials = dict(specials or {})
        if unk_token not in self.vocab:
            self.vocab[unk_token] = max(self.vocab.values(), default=-1) + 1
        self.inverse = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(tok, self.vocab[self.unk_token]) for tok in text.split()]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.inverse.get(int(i), self.unk_token) for i in ids)

    @property
    def vocab_size(self) -> int:
        return max(self.vocab.values(), default=-1) + 1

    @property
    def special_tokens(self) -> dict[str, int]:
        return dict(self._specials)


def tokenizer_byte(**kwargs) -> ByteTokenizer:
    return ByteTokenizer(**kwargs)


def tokenizer_bpe(vocab: Mapping[str, int] | None = None, merges: Sequence[tuple[str, str]] | None = None, **kwargs) -> VocabTokenizer:
    return VocabTokenizer(vocab or {}, **kwargs)


def tokenizer_wordpiece(vocab: Mapping[str, int] | None = None, **kwargs) -> VocabTokenizer:
    return VocabTokenizer(vocab or {}, **kwargs)


def tokenizer_unigram(vocab: Mapping[str, int] | None = None, **kwargs) -> VocabTokenizer:
    return VocabTokenizer(vocab or {}, **kwargs)


def tokenizer_sentencepiece_compat(vocab: Mapping[str, int] | None = None, **kwargs) -> VocabTokenizer:
    return VocabTokenizer(vocab or {}, **kwargs)


def _stack_batch(items: list[Any]) -> Any:
    first = items[0]
    if isinstance(first, Mapping):
        return {k: np.stack([np.asarray(item[k]) for item in items]) for k in first}
    try:
        return np.stack([np.asarray(x) for x in items])
    except ValueError:
        return list(items)


def _axis_size(mesh: Any, axis_name: str) -> int:
    if hasattr(mesh, "axis_size"):
        return int(mesh.axis_size(axis_name))
    if isinstance(mesh, Mapping):
        return int(mesh[axis_name])
    return 1


__all__ = [
    "ByteTokenizer",
    "Dataset",
    "IterableDataset",
    "ShardedDataset",
    "Tokenizer",
    "VocabTokenizer",
    "dataset_batch",
    "dataset_checkpoint",
    "dataset_filter",
    "dataset_interleave",
    "dataset_map",
    "dataset_prefetch",
    "dataset_repeat",
    "dataset_shuffle",
    "dataset_zip",
    "iterable_dataset",
    "sharded_dataset",
    "tokenizer_bpe",
    "tokenizer_byte",
    "tokenizer_sentencepiece_compat",
    "tokenizer_unigram",
    "tokenizer_wordpiece",
]
