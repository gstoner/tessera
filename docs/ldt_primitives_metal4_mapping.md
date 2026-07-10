# LDT primitives → Metal 4 capability mapping (archived)

> **This note has landed and been consolidated.** The current Apple GPU state
> — including the Metal-envelope dispatch + numpy-fallback chain these
> primitives ride — lives in the single Apple backend reference,
> [apple_backend.md](apple_backend.md) (and the kernel + lane inventory,
> [apple_gpu_kernel_inventory.md](apple_gpu_kernel_inventory.md)).
>
> The full historical document (the functional → perf upgrade path for
> `count_nonzero`, `popcount`, `asymmetric_bce`, `masked_categorical`) is
> preserved for provenance at
> [`docs/audit/backend/apple/archive/ldt_primitives_metal4_mapping.md`](audit/backend/apple/archive/ldt_primitives_metal4_mapping.md).
