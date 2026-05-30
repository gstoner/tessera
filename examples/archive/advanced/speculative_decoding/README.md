# Speculative and Tree Decoding

This example adds a Yggdrasil/Medusa/EAGLE-style decoding scaffold for Tessera.
It focuses on the scheduling shape rather than model-specific weights:

- draft tree expansion
- target-model verification
- accepted-token compaction
- decode-lane scheduling metadata

## Quick Start

```bash
python3 examples/advanced/speculative_decoding/demo.py --depth 3 --branching 3
```

## Tessera Mapping

- Graph IR: represent draft branches as a bounded decode tree.
- Schedule IR: batch sibling verification into one target-model pass.
- Tile IR: compact accepted branches and roll KV pages forward.
- Runtime: tune depth/branching per model latency and acceptance rate.
