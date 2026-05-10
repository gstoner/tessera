"""S15 native data pipeline and tokenizer coverage."""

from __future__ import annotations

import numpy as np

import tessera as ts


def test_dataset_combinators_and_checkpoint_restore():
    ds = ts.data.Dataset.synthetic(6).map(lambda x: x + 1).filter(lambda x: x % 2 == 0)
    np.testing.assert_array_equal(ds.to_list(), [2, 4, 6])

    batches = ds.batch(2).to_list()
    np.testing.assert_array_equal(batches[0], [2, 4])
    np.testing.assert_array_equal(batches[1], [6])

    repeated = ds.repeat(2)
    np.testing.assert_array_equal(repeated.to_list(), [2, 4, 6, 2, 4, 6])

    zipped = ds.zip(ts.data.Dataset.synthetic(3, lambda i: i * 10)).to_list()
    assert zipped[0] == (2, 0)

    restored = ds.restore({"cursor": 1})
    np.testing.assert_array_equal(restored.to_list(), [4, 6])
    assert restored.checkpoint()["cursor"] == 1


def test_shuffle_sharding_iterable_and_file_source(tmp_path):
    key = ts.rng.RNGKey.from_seed(42)
    ds = ts.data.Dataset.synthetic(8)
    shuffled_a = ds.shuffle(key).to_list()
    shuffled_b = ds.shuffle(ts.rng.RNGKey.from_seed(42)).to_list()
    assert shuffled_a == shuffled_b
    assert shuffled_a != ds.to_list()

    mesh = ts.NamedMesh(("dp",), (2,))
    shard0 = ts.data.sharded_dataset(ds, mesh, "dp", index=0).to_list()
    shard1 = ts.data.sharded_dataset(ds, mesh, "dp", index=1).to_list()
    assert shard0 == [0, 2, 4, 6]
    assert shard1 == [1, 3, 5, 7]

    stream = ts.data.iterable_dataset(lambda: range(5)).restore({"cursor": 2})
    assert stream.take(10).to_list() == [2, 3, 4]

    path = tmp_path / "source.npz"
    np.savez(path, x=np.arange(3), y=np.arange(3) + 10)
    rows = ts.data.Dataset.sharded_file_source(path).to_list()
    assert rows[0]["x"] == 0 and rows[0]["y"] == 10


def test_tokenizers_round_trip_and_vocab_metadata():
    byte_tok = ts.data.tokenizer_byte(specials={"<pad>": 0})
    text = "hello"
    assert byte_tok.decode(byte_tok.encode(text)) == text
    assert byte_tok.vocab_size == 256
    assert byte_tok.special_tokens["<pad>"] == 0

    vocab = {"hello": 1, "world": 2}
    for factory in (
        ts.data.tokenizer_bpe,
        ts.data.tokenizer_wordpiece,
        ts.data.tokenizer_unigram,
        ts.data.tokenizer_sentencepiece_compat,
    ):
        tok = factory(vocab, specials={"<bos>": 3})
        ids = tok.encode("hello world missing")
        assert ids[:2] == [1, 2]
        assert ids[2] == tok.vocab["<unk>"]
        assert tok.decode(ids).startswith("hello world")
        assert tok.special_tokens["<bos>"] == 3
