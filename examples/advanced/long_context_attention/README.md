# Long-Context Attention Head Specialization

This example sketches a Tessera implementation of retrieval-head vs streaming-head
attention. It is designed for long-context inference where only a subset of heads
need full historical KV access while the remaining heads can use a bounded recent
window plus sink tokens.

## What It Demonstrates

- head classification from lightweight attention statistics
- per-head cache plans: `retrieval`, `streaming`, or `sink_stream`
- memory estimates for full vs specialized KV layouts
- hook points for Tessera schedule lowering and head-wise cache placement

## Quick Start

```bash
python3 examples/advanced/long_context_attention/demo.py --heads 16 --seq-len 131072
```

## Tessera Mapping

- Graph IR: annotate attention heads with `ts.attn.head_role`.
- Schedule IR: split heads into retrieval and streaming groups.
- Tile IR: assign retrieval heads to paged KV blocks; assign streaming heads to
  ring buffers with sink-token prefix blocks.
- Runtime: periodically refresh head roles using sampled attention statistics.
