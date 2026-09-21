# Jet-Nemotron linear-attention compiler slice

The maintained example is a self-contained D=128 linear-attention smoke using
the canonical Tessera API:

```bash
python3 examples/advanced/Jet_nemotron/tests/smoke_linear_attention.py
```

It validates `tessera.ops.linear_attn` against a NumPy oracle and requires
non-empty Graph, Schedule, Tile, and Target IR. This is portable compiler and
numerical evidence; it is not ROCm exact-device or performance evidence.

The former full-model/PostNAS port depended on an unshipped `tessera.stdlib`,
ghost package imports, and placeholder runtime surfaces. It is preserved for
provenance under `archive/examples/advanced/Jet_nemotron/` and is not part of
the active examples contract.
