<!-- MERGE_START: ATLAS_DOCS -->
# Atlas → Tessera Mapping (v0.1) — 2025-09-17

This document maps **ATLAS: Learning to Optimally Memorize the Context at Test Time** to the Tessera programming model.

**Key ideas to capture in Tessera:** sliding-window *Omega* updates over a **deep memory module**, higher-order **feature mappings** for keys/queries (e.g., polynomial), and an **internal optimizer** (e.g., Muon) that updates memory *at test time* (no outer-loop weight updates).

## What Atlas is (from the paper)
- A long-term memory module that **memorizes the context** rather than individual tokens by optimizing an internal objective over a **window of past tokens** (“Omega rule”).  
- Uses **higher-order feature maps** to expand memory capacity.  
- Uses a **locally optimal** internal optimizer; the paper uses **Muon** for second-order-ish updates to the memory state.  
- Introduces **DeepTransformers / SWDT** as strict generalizations of Transformers with sliding windows and deep memory.

## Tessera affordances we’ll use
- **Graph IR**: compose memory stages around model blocks.
- **Schedule IR**: tile sequence into windows; overlap compute/comm; page KV/state.
- **Tile IR**: fused kernels for feature mapping, update (optimizer), and read.
- **Target IR**: MMA/MFMA, TMA/LDS staging, async copy, stream/comm ops.

See paper for details and terminology.  
<!-- MERGE_END: ATLAS_DOCS -->