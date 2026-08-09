---
audit_role: reference
last_updated: 2026-08-09
scope: native multi-node RCCL GIN correctness and performance evidence
---

# RCCL GIN exact-device packet runbook

`tessera-rccl-gin-smoke` is the native launcher boundary for the
`COLLECTIVE-RCCL-ADVANCED-LANES-2026-08-09` GIN/RMA lane. It creates one RCCL
rank per process with `ncclCommInitRank`, registers one symmetric strict-ordering
destination window per rank, and executes a ring of `put_signal`, standalone
`signal`, and `wait_signal` operations. Exact readback proves ordering. The
timed interval measures a batched put/wait ring using both HIP events and
steady host wall time; the emitted packet uses the maximum rank time.

The binary has no MPI ABI dependency. It discovers rank metadata from explicit
`TESSERA_GIN_*` variables, OpenMPI, PMI, or Slurm variables, in that order. The
launcher must provide one process per GPU and a shared filesystem directory for
the RCCL unique ID, phase barriers, and per-rank evidence. `TESSERA_GIN_RUN_ID`
must be unique so a stale rendezvous file cannot satisfy a new run.

Required shared variables:

- `TESSERA_GIN_RENDEZVOUS`: directory visible to every node.
- `TESSERA_GIN_RUN_ID`: unique alphanumeric, `_`, or `-` token.
- `TESSERA_GIN_ARTIFACT_DIGEST`: 64-hex digest of the exact one-sided Target
  artifact.
- `TESSERA_GIN_COMMUNICATOR_DIGEST`: 64-hex topology snapshot digest sealed by
  `communicator_capability_snapshot`.

Optional variables are `TESSERA_GIN_BYTES` (default 1 MiB),
`TESSERA_GIN_WARMUP` (10), `TESSERA_GIN_ITERATIONS` (100), and
`TESSERA_GIN_TIMEOUT_SECONDS` (120). Explicit `TESSERA_GIN_RANK`,
`TESSERA_GIN_WORLD_SIZE`, and `TESSERA_GIN_LOCAL_RANK` override launcher
metadata.

An OpenMPI launch has this shape:

```bash
mpirun -np 2 --map-by ppr:1:node \
  -x TESSERA_GIN_RENDEZVOUS -x TESSERA_GIN_RUN_ID \
  -x TESSERA_GIN_ARTIFACT_DIGEST -x TESSERA_GIN_COMMUNICATOR_DIGEST \
  /path/to/tessera-rccl-gin-smoke
```

A Slurm launch has this shape:

```bash
srun --nodes=2 --ntasks-per-node=1 --gpus-per-task=1 \
  /path/to/tessera-rccl-gin-smoke
```

Rank zero emits `tessera.rccl_gin_packet.v1`. Promotion requires all ranks to
be gfx1151, initialized properties with `host_rma_support=true` and nonzero
`gin_type`, exact readback, nonzero HIP-event time, and both supplied digests.
The packet retains RCCL/HIP versions, host and architecture per rank, message
size, iteration policy, and independent clock values. A missing capability,
single rank, absent launcher metadata, or non-gfx1151 device exits 77 and is a
hardware/access blocker rather than a pass. A nonzero correctness exit is a
failure. Gfx1250 DDA and zero-CU Copy Engine evidence remain separate packets.
