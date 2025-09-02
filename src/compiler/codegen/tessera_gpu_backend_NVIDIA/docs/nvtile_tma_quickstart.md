# NVIDIA TMA Quickstart (Minimal)

This backend's WGMMA demo can optionally use **TMA 2D bulk copies** if you pass non-null TMA descriptors into
`tessera_gpu_wgmma_bf16(...)`. Descriptor creation is host-side and depends on CUDA 12+ APIs.

## Host-side (conceptual)

Use CUDA's tensor map/descriptor APIs (e.g., `cuTensorMapEncodeTiled` via Driver API) to encode a 2D layout for A and B.
Then pass the resulting descriptor device pointers to the kernel launch (or copy them into device memory and pass the device address).

**Fallback:** if you pass `nullptr` for the descriptors, the kernel will perform a simple (non-TMA) staging path.

> This is a minimal demo; production code should set up proper **TMA** descriptors and synchronize with **mbarrier** for pipelined multi-slab copies.
