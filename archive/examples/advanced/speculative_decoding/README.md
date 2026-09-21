# Speculative and Tree Decoding

This archived example is the former Yggdrasil/Medusa/EAGLE-style string
scheduling toy. Maintained speculative-decoding coverage lives in
`examples/advanced/gumiho/`.
It focuses on the scheduling shape rather than model-specific weights:

- draft tree expansion
- target-model verification
- accepted-token compaction
- decode-lane scheduling metadata

## Quick Start

```bash
PYTHONPATH=python python3 archive/examples/advanced/speculative_decoding/demo.py --depth 3 --branching 3
```

## Tessera Mapping

- Graph IR: represent draft branches as a bounded decode tree.
- Schedule IR: batch sibling verification into one target-model pass.
- Tile IR: compact accepted branches and roll KV pages forward.
- Runtime: tune depth/branching per model latency and acceptance rate.
